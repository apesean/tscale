# tscale

新服务器的硬件摸底脚本：先看清楚机器有什么，再决定上面能跑什么（MatterGen、kernel optimization 等）。

## 用法

在服务器上：

```bash
git clone https://github.com/apesean/tscale.git
```

```bash
cd tscale && ./run_all.sh
```

全部跑完大约 3–5 分钟，每块 GPU 再加 1 分钟左右。结果在 `results/<TAG>/`，其中 `report.md` 是合并后的完整报告。

把结果传回来（二选一）：

```bash
git add results && git commit -m "probe results" && git push
```

或者直接把 `results/<TAG>/report.md` 的内容贴进对话。

> 这是一个**公开仓库**。脚本不采集 IP、MAC、序列号，并且默认把报告里的主机名和用户名替换成 `<host>` / `<user>`。
> 提交前仍建议扫一眼 `report.md`。

## 脚本做了什么

| 步骤 | 文件 | 内容 | 依赖 |
| --- | --- | --- | --- |
| 1 | `probe/01_system.sh` | 只读清单：OS/内核、CPU 指令集、内存/NUMA、磁盘、GPU 与驱动、CUDA 工具链、编译器、perf/eBPF 等调优工具、内核头文件与配置、权限与资源限制、Python/容器环境 | bash |
| 2 | `probe/02_bench_cpu.py` | 单核速度、多进程扩展性、内存带宽（编译 `stream_mini.c`）、BLAS、磁盘顺序/随机/小文件 | python3 标准库；gcc、numpy 可选 |
| 3 | `probe/03_bench_gpu.py` | 每块 GPU：GEMM (fp32/TF32/fp16/bf16)、主机↔显存传输、图网络训练步 proxy、持续满载下的温度/功耗/降频 | PyTorch |
| 4 | `probe/04_mattergen_ready.py` | 对照 MatterGen 的依赖逐项给出 PASS/WARN/FAIL：驱动版本、GPU 架构与 torch 2.2.1+cu121 是否兼容、显存、uv/Python 3.10、git-lfs、磁盘、到 GitHub / PyPI / PyTorch / PyG / Hugging Face 的连通性与下载速度 | python3 标准库 |

所有步骤都不需要 root，不修改系统配置。任何一步缺依赖都会跳过并在报告里说明，不会中断。
磁盘测试只在目标目录写一个 1 GiB 的临时文件，结束后删除；剩余空间不足 4 GiB 的目录不会被碰。

## 常用选项

```bash
QUICK=1 ./run_all.sh
```

快速模式，约 1 分钟，数字不够准，适合先确认脚本能跑通。

```bash
BENCH_DIRS="/data /scratch" ./run_all.sh
```

额外测试这些目录所在的磁盘（数据盘、scratch 盘）。

```bash
TORCH_PYTHON=/path/to/env/bin/python ./run_all.sh
```

GPU 基准需要 PyTorch。如果机器上已有装了 torch 的环境，用这个变量指过去。

```bash
SETUP_TORCH=1 ./run_all.sh
```

机器上没有 torch 时，在 `.venv-probe/` 里装一份（下载约 2–3 GB）。这个环境只用于基准，和之后的 MatterGen 环境无关，用完可以删。

其他：`FORCE_GPU=1`（GPU 上已有任务时也照样测，默认会跳过忙碌的 GPU）、`NO_NETWORK=1`（不做外网探测）、`NO_REDACT=1`（不脱敏）、`TAG=name`（自定义结果目录名）。

## 说明

- GPU 的 “GNN proxy” 是 gather + MLP + scatter 的小型消息传递网络，用来横向比较 GPU 在图网络负载上的表现，**不是** MatterGen 本身的速度。真实数字要等 MatterGen 装好后再测。
- `04_mattergen_ready.py` 里标了 `[heuristic]` 的阈值（内存、磁盘、显存分档）是经验值，不是 MatterGen 官方要求。
- MatterGen 在 Linux 上固定 `torch==2.2.1` + CUDA 12.1。这个版本的 torch 最高只支持到 compute capability 9.0 的 GPU；更新架构的显卡需要改依赖，脚本会明确报出来。
