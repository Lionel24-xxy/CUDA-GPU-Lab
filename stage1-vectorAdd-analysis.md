# Stage1 CUDA-VectorAdd 性能分析

显卡：NVIDIA GeForce RTX 3050 4GB Laptop GPU

环境：Windows11 WSL2 Ubuntu

## 1. 生成报告

vectorAdd核心代码（使用Nvidia官方示例）

```cpp
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }

    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Use unified memory to allocate buffers
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 64;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // Wait for the kernel to complete execution
    cudaDeviceSynchronize();

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);

    return 0;
```

先用nsys找到最热点的kernel，当前用例仅有一个kernel（更具体参数见附录）

```bash
# 在 build 目录下
nsys profile -o report ./vectorAdd
nsys stats -r cuda_gpu_kern_sum report.nsys-rep
```

运行结果：

```bash
CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                   Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  --------------------------------------
    100.0             5952          1    5952.0    5952.0      5952      5952          0.0  vecAdd(float *, float *, float *, int)
```

之后使用ncu对单kernel进行深入分析

```bash
ncu --set full ./vectorAdd
```

## 2. 分析

### 一句话结论

当前配置下的kernel负载太小了，GPU无法打满，当前数据只有1024，block size设置为256，Grid size只有4，远小于20的最大值，硬件允许每个SM同时驻留48个warp，但是目前每个SM只有约7个warp，无法通过 warp 级并行隐藏 global memory 访问延迟

### 主要指标

| Metric Name             | Metric Value |
| ----------------------- | ------------ |
| Memory Throughput       | 1.70%        |
| Compute (SM) Throughput | 0.20%        |
| Theoretical Occupancy   | 100%         |
| Achieved Occupancy      | 15.74%       |
| Waves Per SM            | 0.03         |
| No Eligible             | 98.72%       |
| Long Scoreboard stall   | 79.11%       |

### 3. 进一步实验

将数据量从1024增加至5000000，指标变化：

| Metric Name             | Metric Value（1024） | Metric Value（5000000） |
| ----------------------- | -------------------- | ----------------------- |
| Memory Throughput       | 1.70%                | 47.99%                  |
| Compute (SM) Throughput | 0.20%                | 3.13%                   |
| Theoretical Occupancy   | 100%                 | 100%                    |
| Achieved Occupancy      | 15.74%               | 96.15%                  |
| Waves Per SM            | 0.03                 | 162.77                  |
| No Eligible             | 98.72%               | 98.44%                  |
| Long Scoreboard stall   | 79.11%               | 98.0%                   |

可以看到Achieved Occupancy有显著的提升，可以看到当前kernel的主要瓶颈来自带宽，Long Scoreboard stall的升高看起来比较异常，实际原因是源码用的是 `cudaMallocManaged`（Unified Memory）。在 WSL2 的 WDDM 驱动模型下，GPU 访问 Unified Memory 页面时，**不是直接走显存，而是通过 page fault 机制从系统内存迁移或访问数据**。这个路径的延迟远高于普通 `cudaMalloc` 的显存访问，而且**每次 page fault 的处理时间不可预测**。

## 附录

### nsys指令参数

#### `nsys profile`：采集阶段的参数

**`-o` / `--output`：指定输出文件名**  
你用的 `-o report`，就会生成 `report.nsys-rep`。如果不指定，默认会以 `<程序名>_<时间戳>` 命名，容易产生大量文件，所以**养成指定 `-o` 的习惯是好的**。

**`-t` / `--trace`：选择要追踪的活动类型（最核心的过滤参数）**  
这是控制采集范围的开关，用逗号分隔多个目标，且**逗号后不能有空格**。默认会追踪 `cuda,opengl,nvtx,osrt`，但大部分计算任务不需要 OpenGL，显式指定可以减小文件体积、降低开销。

**`--stats=true`：采集完立即输出文本摘要**  
加上这个参数，`nsys` 会在采集结束后**直接在终端打印**默认的几张统计表（包括你熟悉的 `cuda_gpu_kern_sum`），省去再手动跑一次 `nsys stats` 的步骤。适合快速查看结果。

**`--force-overwrite=true`：允许覆盖同名报告**  
如果不加，当 `report.nsys-rep` 已存在时，`nsys` 会报错退出。在反复调试的场景下，加上它可以避免手动删除旧文件的麻烦。

**`--duration` / `-d` 与 `--delay` / `-y`：控制采集的时间窗口**  
对于长期运行的服务或训练脚本，不需要从头录到尾

- `--duration 10`：只采集程序启动后的**前 10 秒**。

- `--delay 5 --duration 10`：**跳过前 5 秒**，然后采集 10 秒。适合跳过模型初始化、预热阶段，只抓取稳定运行状态。

**`--gpu-metrics-devices=all`：采集 GPU 硬件指标**  
这是一个进阶选项，可以采集 SM 占用率、DRAM 吞吐等硬件计数器，对判断 compute-bound / memory-bound 有直接帮助。但**开销较大**，且需要 `nsys` 版本支持（2023.3+）。

#### `nsys stats`：解读阶段的参数

**`-r` / `--report`：指定要生成的报告类型（核心参数）**  
你用的 `-r cuda_gpu_kern_sum` 就是指定生成“GPU Kernel 汇总”表。`nsys` 支持多种报告，**最常用的几个**是：

| 报告类型 (`-r` 后的值)      | 关键列                                              |
| --------------------------- | --------------------------------------------------- |
| **`cuda_gpu_kern_sum`**     | Name, Time (%), Total Time, Instances               |
| **`cuda_api_sum`**          | cudaMalloc、cudaMemcpy 等 API 的耗时与次数          |
| **`cuda_gpu_mem_time_sum`** | Memcpy HtoD / DtoH / DtoD 的总时间                  |
| **`osrt_sum`**              | 系统调用（如 poll、ioctl）的耗时，用于诊断 CPU 瓶颈 |
| **`nvtx_sum`**              | 需代码中嵌入 NVTX 标记                              |

**`--format csv`：输出为 CSV 格式**  
便于用 `awk`、`python` 或 Excel 做进一步处理。例如，提取耗时前 10 的 kernel：`nsys stats -r cuda_gpu_kern_sum --format csv report.nsys-rep | head -15`。默认输出是对齐的文本表格，适合直接阅读。

### ncu指令参数

**`-k` / `--kernel-name`：按内核名称过滤（最常用）**  
支持精确匹配和正则表达式，是定位目标内核的首选方式。

**`-s` / `--launch-skip` 与 `-c` / `--launch-count`：按启动次数过滤**  
当目标内核在程序中会被**多次调用**时（例如训练循环中的迭代），这两个参数用于**跳过前 N 次，只分析后续的 M 次**，以捕获稳定运行阶段的数据。

**`--kernel-id`：按更复杂的标识符过滤**  
可以按上下文、流、名称和调用次数的组合来精确定位内核，类似旧版 `nvprof` 的过滤方式。

**`--nvtx` 与 `--nvtx-include`：按代码中的NVTX标记过滤**  
如果你的代码中插入了NVTX标记（用于标注代码区间），可以用这些参数只分析特定标记区间内的内核。

**`--set <集合名>`：使用预定义的指标集合（最常用）**  
这是平衡采集深度与开销的核心参数，`ncu` 提供了多种预设集合。

| 集合 (`--set`) | 适用场景                                   | 包含的核心指标                                             |
| -------------- | ------------------------------------------ | ---------------------------------------------------------- |
| **`basic`**    | **推荐的起点**，快速了解内核概况           | LaunchStats, Occupancy, SpeedOfLight, WorkloadDistribution |
| **`full`**     | **深度分析**，指标最全面，但开销极大       | 几乎包含所有指标（Nsight Compute 2026.3 中约9000个）       |
| `memory`       | 重点分析**内存子系统**（带宽、缓存命中率） | 内存工作负载相关指标                                       |
| `compute`      | 重点分析**计算吞吐**（SM利用率、指令发射） | 计算工作负载相关指标                                       |
| `occupancy`    | 专项分析**占用率**瓶颈                     | Occupancy相关指标                                          |
| `roofline`     | 生成**Roofline模型**数据，判断性能上限     | SpeedOfLight + Roofline图表                                |
| `launch`       | 分析**内核启动配置**（Grid/Block大小）     | LaunchStatistics                                           |

**`--section <节名>`：采集指定的指标“章节”**  
比 `--set` 更精细的控制，可以只采集你关心的几个分析章节。

```bash
# 只采集SpeedOfLight和Occupancy两个章节
ncu --section SpeedOfLight --section Occupancy ./program
```

常用章节包括：`SpeedOfLight`, `Occupancy`, `MemoryWorkloadAnalysis`, `ComputeWorkloadAnalysis`, `SchedulerStatistics`, `WarpStateStatistics` 等。

**`--metrics <指标1,指标2,...>`：直接指定底层指标名**  
用于采集非常具体的硬件计数器，需要知道完整的指标名称（如 `sm__throughput.avg.pct_of_peak_sustained_elapsed`）。这是高级用法，通常在需要验证某个特定假设时使用。

**`--list-sets` / `--list-sections` / `--list-metrics`：查看可用选项**  
如果不确定有哪些集合、章节或指标可用，这些命令会列出当前环境下所有支持的选项。

**`-o` / `--export`：导出报告文件**  
将分析结果保存为 `.ncu-rep` 文件，便于之后用 **Nsight Compute GUI** 打开进行交互式深度分析。

**`--print-summary <格式>`：在终端打印摘要**  
不保存文件，直接在命令行输出简要结果，适合快速查看。

**`--csv`：输出为CSV格式**  
便于脚本处理和数据分析。
