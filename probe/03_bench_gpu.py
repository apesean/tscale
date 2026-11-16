#!/usr/bin/env python3
"""03_bench_gpu.py - GPU 基准（需要 PyTorch）。

对每块 GPU 依次测试:
  1. GEMM 吞吐 (fp32 / fp32+TF32 / fp16 / bf16)        -> TFLOP/s
  2. 主机<->显存 传输带宽 (pinned memory)               -> GB/s（PCIe 链路是否正常）
  3. GNN 训练步 proxy（gather + MLP + scatter，近似 MatterGen/GemNet 这类
     图网络的算子组合，不是 MatterGen 本身）             -> structures/s, 峰值显存
  4. 持续满载 N 秒并采样温度/功耗/频率                   -> 散热与降频检查

如果某块 GPU 已经有别人的任务在跑（显存或利用率偏高），默认跳过，用 --force 强制。

用法:
    python 03_bench_gpu.py [--out gpu.json] [--gpus 0,1] [--burn 20] [--quick] [--force]
    python 03_bench_gpu.py --device cpu --quick      # 仅用于自检脚本逻辑
"""
from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys
import threading
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # 让 torch 的编号与 nvidia-smi 一致

try:
    import torch
    import torch.nn as nn
except Exception as e:  # noqa
    print("PyTorch 不可用 (%s: %s)" % (type(e).__name__, e))
    print("跳过 GPU 基准。可以: (a) TORCH_PYTHON=/path/to/env/bin/python ./run_all.sh")
    print("            或 (b) SETUP_TORCH=1 ./run_all.sh  (会在 .venv-probe 里 pip install torch，约 2-3 GB)")
    sys.exit(3)


def smi_query(fields, index=None):
    cmd = ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
    if index is not None:
        cmd += ["-i", str(index)]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        out, _ = p.communicate(timeout=10)
        if p.returncode != 0:
            return None
        return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines()]
    except Exception:
        return None


def to_float(s):
    try:
        return float(s)
    except Exception:
        return None


def smi_index_for(torch_index):
    """torch device index -> nvidia-smi index (None if it cannot be determined)."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not vis:
        return torch_index
    parts = [p.strip() for p in vis.split(",") if p.strip()]
    if all(p.isdigit() for p in parts) and torch_index < len(parts):
        return int(parts[torch_index])
    return None


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def is_oom(e):
    return "out of memory" in str(e).lower()


# ------------------------------------------------------------------ benchmarks
def bench_gemm(dev, quick):
    n = 1024 if dev.type == "cpu" else (4096 if quick else 8192)
    out = {"n": n}
    variants = [("fp32", torch.float32, False)]
    if dev.type == "cuda":
        variants += [("fp32_tf32", torch.float32, True), ("fp16", torch.float16, False), ("bf16", torch.bfloat16, False)]
    for name, dtype, tf32 in variants:
        try:
            if dev.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = tf32
            a = torch.randn(n, n, device=dev, dtype=dtype)
            b = torch.randn(n, n, device=dev, dtype=dtype)
            for _ in range(2):
                (a @ b)
            sync(dev)
            iters, dt = 0, 0.0
            t0 = time.perf_counter()
            while dt < (0.5 if quick else 2.0):
                for _ in range(5):
                    (a @ b)
                sync(dev)
                iters += 5
                dt = time.perf_counter() - t0
            out[name + "_TFLOPS"] = round(2.0 * n ** 3 * iters / dt / 1e12, 2)
            del a, b
        except Exception as e:  # noqa
            out[name + "_error"] = ("OOM" if is_oom(e) else "%s: %s" % (type(e).__name__, str(e)[:120]))
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    print("  GEMM n=%d: %s" % (n, ", ".join("%s=%s" % (k, v) for k, v in out.items() if k != "n")))
    return out


def bench_transfer(dev, quick):
    if dev.type != "cuda":
        return {"skipped": "cpu"}
    mb = 256 if quick else 1024
    out = {"size_MiB": mb}
    try:
        host = torch.empty(mb * 1024 * 1024, dtype=torch.uint8).pin_memory()
        host.fill_(1)
        g = host.to(dev, non_blocking=True)
        sync(dev)
        best_up, best_down = 1e9, 1e9
        for _ in range(3):
            t = time.perf_counter()
            g = host.to(dev, non_blocking=True)
            sync(dev)
            best_up = min(best_up, time.perf_counter() - t)
            t = time.perf_counter()
            host.copy_(g)
            sync(dev)
            best_down = min(best_down, time.perf_counter() - t)
        out["host_to_gpu_GBps"] = round(mb / 1024.0 / best_up * 1.0737, 2)
        out["gpu_to_host_GBps"] = round(mb / 1024.0 / best_down * 1.0737, 2)
        del g, host
    except Exception as e:  # noqa
        out["error"] = "%s: %s" % (type(e).__name__, str(e)[:120])
    print("  transfer: %s" % out)
    return out


class _Layer(nn.Module):
    def __init__(self, h):
        super(_Layer, self).__init__()
        self.msg = nn.Sequential(nn.Linear(3 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.upd = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.norm = nn.LayerNorm(h)

    def forward(self, x, e, src, dst):
        m = self.msg(torch.cat([x[src], x[dst], e], dim=-1))
        agg = torch.zeros(x.shape[0], m.shape[1], dtype=m.dtype, device=x.device).index_add_(0, dst, m)
        x = self.norm(x + self.upd(torch.cat([x.to(m.dtype), agg], dim=-1)))
        return x, e + m


class _GNN(nn.Module):
    def __init__(self, h, layers):
        super(_GNN, self).__init__()
        self.layers = nn.ModuleList([_Layer(h) for _ in range(layers)])
        self.head = nn.Linear(h, 3)

    def forward(self, x, e, src, dst):
        for layer in self.layers:
            x, e = layer(x, e, src, dst)
        return self.head(x)


def bench_gnn(dev, quick):
    """Train-step throughput of a small message-passing net on random crystal-sized graphs."""
    atoms, neigh, hidden, layers = 20, 32, 256, 4
    batches = [4] if dev.type == "cpu" else ([16, 64] if quick else [16, 64, 256])
    modes = [("fp32", None)]
    if dev.type == "cuda":
        modes.append(("amp_bf16", torch.bfloat16))
    out = {"atoms_per_structure": atoms, "neighbors": neigh, "hidden": hidden, "layers": layers, "runs": []}
    for mode, amp_dtype in modes:
        for bs in batches:
            run = {"mode": mode, "batch_structures": bs}
            try:
                torch.manual_seed(0)
                model = _GNN(hidden, layers).to(dev)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
                n = bs * atoms
                # each atom gets `neigh` random neighbours inside its own structure
                base = (torch.arange(n, device=dev) // atoms * atoms).repeat_interleave(neigh)
                src = base + torch.randint(0, atoms, (n * neigh,), device=dev)
                dst = torch.arange(n, device=dev).repeat_interleave(neigh)
                x = torch.randn(n, hidden, device=dev)
                e = torch.randn(n * neigh, hidden, device=dev)
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(dev)

                def step():
                    opt.zero_grad(set_to_none=True)
                    if amp_dtype is not None:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            loss = model(x, e, src, dst).float().pow(2).mean()
                    else:
                        loss = model(x, e, src, dst).pow(2).mean()
                    loss.backward()
                    opt.step()

                for _ in range(3):
                    step()
                sync(dev)
                iters, dt = 0, 0.0
                t0 = time.perf_counter()
                while dt < (0.5 if quick else 3.0):
                    step()
                    iters += 1
                    sync(dev)
                    dt = time.perf_counter() - t0
                run["steps_per_s"] = round(iters / dt, 2)
                run["structures_per_s"] = round(iters * bs / dt, 1)
                if dev.type == "cuda":
                    run["peak_mem_GiB"] = round(torch.cuda.max_memory_allocated(dev) / 2.0 ** 30, 2)
                del model, opt, x, e, src, dst
            except Exception as ex:  # noqa
                run["error"] = "OOM" if is_oom(ex) else "%s: %s" % (type(ex).__name__, str(ex)[:160])
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            out["runs"].append(run)
            print("  GNN proxy %-8s bs=%-4d %s" % (mode, bs, {k: v for k, v in run.items() if k not in ("mode", "batch_structures")}))
    return out


def burn(dev, seconds, smi_index):
    """Sustained load while sampling nvidia-smi once per second."""
    if dev.type != "cuda" or seconds <= 0:
        return {"skipped": True}
    full = ["temperature.gpu", "power.draw", "utilization.gpu", "clocks.current.graphics", "clocks_throttle_reasons.active"]
    basic = ["temperature.gpu", "power.draw", "utilization.gpu"]
    fields = None
    if smi_index is not None:
        for cand in (full, basic):
            if smi_query(cand, smi_index):
                fields = cand
                break
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            rows = smi_query(fields, smi_index)
            if rows:
                samples.append(rows[0])
            stop.wait(1.0)

    th = None
    if fields:
        th = threading.Thread(target=sampler)
        th.daemon = True
        th.start()
    n = 4096
    a = torch.randn(n, n, device=dev, dtype=torch.float16)
    b = torch.randn(n, n, device=dev, dtype=torch.float16)
    rates = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        t = time.perf_counter()
        for _ in range(20):
            (a @ b)
        sync(dev)
        rates.append(20 * 2.0 * n ** 3 / (time.perf_counter() - t) / 1e12)
    stop.set()
    if th:
        th.join(timeout=5)
    k = max(1, len(rates) // 5)
    out = {
        "seconds": seconds,
        "tflops_first_fifth": round(sum(rates[:k]) / k, 2),
        "tflops_last_fifth": round(sum(rates[-k:]) / k, 2),
    }
    if samples:
        col = lambda i: [v for v in (to_float(s[i]) for s in samples) if v is not None]  # noqa
        temps, power, util = col(0), col(1), col(2)
        if temps:
            out["temp_C_start_end_max"] = [temps[0], temps[-1], max(temps)]
        if power:
            out["power_W_avg_max"] = [round(sum(power) / len(power), 1), max(power)]
        if util:
            out["util_pct_avg"] = round(sum(util) / len(util), 1)
        if len(fields) > 3:
            clocks = col(3)
            if clocks:
                out["clock_MHz_min_max"] = [min(clocks), max(clocks)]
            out["throttle_reasons_seen"] = sorted(set(s[4] for s in samples))
    else:
        out["note"] = "nvidia-smi sampling unavailable"
    print("  burn %ds: %s" % (seconds, out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpus", default=None, help="comma separated torch device indices (default: all)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--burn", type=int, default=20, help="seconds of sustained load per GPU (0 = skip)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true", help="benchmark GPUs even if they look busy")
    args = ap.parse_args()

    result = {
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "python": sys.version.split()[0],
        "gpus": [],
    }
    try:
        result["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    print("torch %s (CUDA build %s), cuda available: %s" % (result["torch"], result["torch_cuda_build"], result["cuda_available"]))

    if args.device == "cpu":
        dev = torch.device("cpu")
        g = {"index": "cpu", "name": "CPU smoke test", "threads": torch.get_num_threads()}
        g["gemm"] = bench_gemm(dev, True)
        g["gnn_proxy"] = bench_gnn(dev, True)
        result["gpus"].append(g)
    elif not torch.cuda.is_available():
        result["error"] = "torch.cuda.is_available() is False (驱动缺失 / torch 是 CPU 版 / 驱动与 torch 的 CUDA 版本不匹配)"
        print(result["error"])
    else:
        try:
            result["torch_arch_list"] = torch.cuda.get_arch_list()
        except Exception:
            result["torch_arch_list"] = []
        count = torch.cuda.device_count()
        wanted = [int(i) for i in args.gpus.split(",")] if args.gpus else list(range(count))
        for i in wanted:
            dev = torch.device("cuda:%d" % i)
            props = torch.cuda.get_device_properties(i)
            cap = torch.cuda.get_device_capability(i)
            smi_i = smi_index_for(i)
            g = {
                "index": i,
                "nvidia_smi_index": smi_i,
                "name": props.name,
                "total_mem_GiB": round(props.total_memory / 2.0 ** 30, 1),
                "compute_capability": "%d.%d" % cap,
                "arch_supported_by_this_torch": (("sm_%d%d" % cap) in result["torch_arch_list"])
                if result["torch_arch_list"] else None,
            }
            print("\n=== GPU %d: %s, %.1f GiB, sm_%d%d ===" % (i, props.name, g["total_mem_GiB"], cap[0], cap[1]))
            busy = None
            if smi_i is not None:
                rows = smi_query(["memory.used", "memory.total", "utilization.gpu"], smi_i)
                if rows:
                    used, total, util = [to_float(v) for v in rows[0]]
                    g["before"] = {"mem_used_MiB": used, "util_pct": util}
                    if used is not None and total and (used / total > 0.3 or (util or 0) > 50):
                        busy = "memory %.0f/%.0f MiB, util %s%%" % (used, total, util)
            if busy and not args.force:
                g["skipped"] = "GPU looks busy (%s); rerun with --force to benchmark anyway" % busy
                print("  SKIP: " + g["skipped"])
                result["gpus"].append(g)
                continue
            try:
                g["gemm"] = bench_gemm(dev, args.quick)
                g["transfer"] = bench_transfer(dev, args.quick)
                g["gnn_proxy"] = bench_gnn(dev, args.quick)
                g["burn"] = burn(dev, 5 if args.quick else args.burn, smi_i)
            except Exception as e:  # noqa
                g["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
                print("  ERROR: " + g["error"])
            torch.cuda.empty_cache()
            result["gpus"].append(g)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print("wrote " + args.out)


if __name__ == "__main__":
    main()
