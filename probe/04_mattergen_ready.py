#!/usr/bin/env python3
"""04_mattergen_ready.py - 检查这台机器能不能直接装/跑 MatterGen。

依据 microsoft/mattergen 的 README 与 pyproject.toml（2026-11 的 main 分支）:
  * Linux 上固定 torch==2.2.1 + CUDA 12.1 (cu121)，PyG 扩展用 cp310 / linux_x86_64 预编译轮子
  * 推荐用 uv 建 Python 3.10 环境
  * 权重与数据集走 Git LFS，默认也会从 Hugging Face 下载

只用标准库 (>=3.6)，只读，除了几个 HTTPS 探测请求外不做任何改动。
标 [heuristic] 的阈值是经验值，不是 MatterGen 官方要求。
"""
from __future__ import print_function

import argparse
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time

try:
    from urllib.request import Request, urlopen
except ImportError:  # pragma: no cover
    print("needs python3")
    sys.exit(2)

CHECKS = []
# torch 2.2.1+cu121 的官方轮子只编译了这些架构；更新的显卡 (如 Blackwell / sm_120) 不在其中
TORCH221_CU121_ARCHS = ((5, 0), (6, 0), (7, 0), (7, 5), (8, 0), (8, 6), (9, 0))
MIN_DRIVER_CUDA121 = (525, 60, 13)  # NVIDIA: CUDA 12.x 在 Linux 上的最低驱动版本


def add(level, name, detail):
    CHECKS.append({"level": level, "check": name, "detail": detail})
    print("[%-4s] %-28s %s" % (level, name, detail))


def sh(cmd, timeout=15):
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out.strip()
    except Exception as e:  # noqa
        return 127, "%s" % e


def vtuple(s):
    out = []
    for part in s.strip().split("."):
        digits = "".join(c for c in part if c.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def check_platform():
    machine = platform.machine()
    if sys.platform.startswith("linux") and machine == "x86_64":
        add("PASS", "platform", "Linux x86_64")
    elif sys.platform.startswith("linux"):
        add("FAIL", "platform", "Linux %s: PyG 的预编译轮子只有 linux_x86_64，需要自己从源码编译" % machine)
    else:
        add("WARN", "platform", "%s %s: README 的安装命令假设 Linux + CUDA" % (sys.platform, machine))
    libc = platform.libc_ver()
    if libc[0] == "glibc" and vtuple(libc[1]) < (2, 17):
        add("FAIL", "glibc", "%s < 2.17 (manylinux2014 轮子装不上)" % libc[1])
    else:
        add("PASS", "glibc", "%s %s" % libc)


def check_gpu():
    if not shutil.which("nvidia-smi"):
        add("FAIL", "nvidia gpu", "找不到 nvidia-smi：没有 NVIDIA GPU 或没装驱动。MatterGen 只能在 CPU 上极慢地跑")
        return []
    rc, out = sh(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    if rc != 0 or not out:
        add("FAIL", "nvidia gpu", "nvidia-smi 运行失败: %s" % out[:200])
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append({"index": parts[0], "name": parts[1], "mem_MiB": float(parts[2]), "driver": parts[3]})
    rc, cap = sh(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if rc == 0:
        for g, c in zip(gpus, cap.splitlines()):
            g["compute_cap"] = c.strip()
    add("PASS", "nvidia gpu", "%d 块: %s" % (len(gpus), "; ".join("%s (%.0f GiB)" % (g["name"], g["mem_MiB"] / 1024) for g in gpus)))

    drv = vtuple(gpus[0]["driver"])
    if drv >= MIN_DRIVER_CUDA121:
        add("PASS", "driver for CUDA 12.1", "%s >= 525.60.13" % gpus[0]["driver"])
    else:
        add("FAIL", "driver for CUDA 12.1", "%s < 525.60.13，cu121 的 torch 用不了，需要升级驱动 (要 root)" % gpus[0]["driver"])

    for g in gpus:
        c = g.get("compute_cap")
        label = "gpu%s arch vs torch 2.2.1" % g["index"]
        if not c:
            if "RTX 50" in g["name"] or "RTX PRO 6000 Blackwell" in g["name"]:
                add("FAIL", label, "%s 看起来是 Blackwell 架构，torch 2.2.1+cu121 不支持" % g["name"])
            else:
                add("WARN", label, "驱动不支持查询 compute_cap，无法自动判断 (%s)" % g["name"])
        elif any(b[0] == vtuple(c)[0] and b[1] <= vtuple(c)[1] for b in TORCH221_CU121_ARCHS):
            add("PASS", label, "compute capability %s 在 torch 2.2.1+cu121 支持范围内" % c)
        elif vtuple(c) > (9, 0):
            add("FAIL", label, "compute capability %s 比 torch 2.2.1+cu121 支持的最高架构 (9.0) 新；"
                               "需要换更新的 torch 并重新匹配/编译 PyG 扩展，MatterGen 的固定依赖要改" % c)
        else:
            add("WARN", label, "compute capability %s 不在 torch 2.2.1+cu121 的预编译架构列表里" % c)

        gib = g["mem_MiB"] / 1024.0
        label = "gpu%s memory [heuristic]" % g["index"]
        if gib < 8:
            add("WARN", label, "%.0f GiB：生成时只能用很小的 batch_size" % gib)
        elif gib < 24:
            add("PASS", label, "%.0f GiB：生成/评估可用，训练或微调要减小 batch 并加大梯度累积" % gib)
        else:
            add("PASS", label, "%.0f GiB：生成、微调都宽裕 (README: batch 512 单卡放不下，默认靠梯度累积)" % gib)
    return gpus


def check_tools():
    py310 = shutil.which("python3.10")
    uv = shutil.which("uv")
    conda = shutil.which("conda") or shutil.which("mamba") or shutil.which("micromamba")
    if uv:
        add("PASS", "uv", "%s (可以自己下载 Python 3.10)" % sh([uv, "--version"])[1])
    else:
        add("WARN", "uv", "未安装。无需 root: curl -LsSf https://astral.sh/uv/install.sh | sh")
    if py310:
        add("PASS", "python 3.10", py310)
    elif uv or conda:
        add("PASS", "python 3.10", "系统没有 python3.10，但可以用 %s 创建" % ("uv" if uv else os.path.basename(conda)))
    else:
        add("WARN", "python 3.10", "系统没有 python3.10，也没有 uv/conda；先装 uv")
    if shutil.which("git"):
        add("PASS", "git", sh(["git", "--version"])[1])
    else:
        add("FAIL", "git", "未安装")
    rc, out = sh(["git", "lfs", "version"])
    if rc == 0:
        add("PASS", "git-lfs", out.splitlines()[0])
    else:
        add("WARN", "git-lfs", "未安装：仓库里的权重/数据集拉不下来 (权重可以改从 Hugging Face 下载)。"
                               "sudo apt install git-lfs，或无 root 时下载官方二进制放到 ~/.local/bin")


def check_resources():
    try:
        with open("/proc/meminfo") as f:
            kb = int(f.readline().split()[1])
        gib = kb / 1048576.0
        add("PASS" if gib >= 30 else "WARN", "system RAM [heuristic]", "%.0f GiB%s" % (gib, "" if gib >= 30 else " (建议 >= 32 GiB)"))
    except Exception:
        pass
    seen = set()
    for label, path in (("home", os.path.expanduser("~")), ("cwd", os.getcwd())):
        try:
            st = os.statvfs(path)
        except OSError:
            continue
        key = (st.f_fsid if hasattr(st, "f_fsid") else path, st.f_blocks)
        if key in seen:
            continue
        seen.add(key)
        free = st.f_bavail * st.f_frsize / 2.0 ** 30
        # torch+CUDA 运行库的虚拟环境约 6-8 GiB，再加 LFS 权重/数据集与 HF 缓存
        add("PASS" if free >= 50 else "WARN", "disk free %s [heuristic]" % label,
            "%.0f GiB%s" % (free, "" if free >= 50 else " (建议 >= 50 GiB：虚拟环境 + 权重 + 数据集 + 缓存)"))
    try:
        st = os.statvfs("/dev/shm")
        shm = st.f_blocks * st.f_frsize / 2.0 ** 30
        add("PASS" if shm >= 2 else "WARN", "/dev/shm", "%.1f GiB%s" % (shm, "" if shm >= 2 else " (DataLoader 多进程可能不够；容器里常见)"))
    except OSError:
        pass


def probe_url(url, timeout, nbytes=0):
    """Returns (ok, detail). With nbytes>0 downloads that many bytes and reports speed."""
    req = Request(url, headers={"User-Agent": "tscale-probe/1.0"})
    if nbytes:
        req.add_header("Range", "bytes=0-%d" % (nbytes - 1))
    t = time.time()
    try:
        resp = urlopen(req, timeout=timeout)
        first = time.time() - t
        if not nbytes:
            resp.read(1024)
            return True, "HTTP %s, %.0f ms" % (resp.getcode(), first * 1000)
        got = 0
        t = time.time()
        while got < nbytes and time.time() - t < 30:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            got += len(chunk)
        dt = max(time.time() - t, 1e-6)
        return True, "%.1f MiB in %.1fs = %.1f MiB/s" % (got / 2.0 ** 20, dt, got / 2.0 ** 20 / dt)
    except Exception as e:  # noqa
        code = getattr(e, "code", None)
        if code and code < 500 and code not in (403, 407):
            return True, "HTTP %s (reachable), %.0f ms" % (code, (time.time() - t) * 1000)
        return False, "%s: %s" % (type(e).__name__, str(e)[:100])


def check_network(timeout):
    targets = [
        ("github.com", "https://github.com/microsoft/mattergen", True),
        ("pypi.org", "https://pypi.org/simple/mattergen/", True),
        ("download.pytorch.org", "https://download.pytorch.org/whl/cu121/", True),
        ("data.pyg.org", "https://data.pyg.org/whl/torch-2.2.0+cu121.html", True),
        ("huggingface.co", "https://huggingface.co/microsoft/mattergen", True),
        ("astral.sh (uv)", "https://astral.sh/uv/install.sh", False),
        ("hf-mirror.com (镜像)", "https://hf-mirror.com/microsoft/mattergen", False),
        ("pypi 清华镜像", "https://pypi.tuna.tsinghua.edu.cn/simple/mattergen/", False),
    ]
    reach = {}
    for name, url, required in targets:
        ok, detail = probe_url(url, timeout)
        reach[name] = ok
        add("PASS" if ok else ("FAIL" if required else "INFO"), "net " + name, detail)
    if reach.get("download.pytorch.org"):
        url = "https://download.pytorch.org/whl/cu121/torch-2.2.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
        ok, detail = probe_url(url, timeout, nbytes=16 << 20)
        add("INFO", "torch wheel download speed", detail + " (整个轮子约 2.2 GiB)" if ok else detail)
    if not reach.get("huggingface.co") and reach.get("hf-mirror.com (镜像)"):
        add("INFO", "hf workaround", "huggingface.co 不通但镜像可用: export HF_ENDPOINT=https://hf-mirror.com")
    if not reach.get("pypi.org") and reach.get("pypi 清华镜像"):
        add("INFO", "pypi workaround", "pypi.org 不通但清华镜像可用: export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-network", action="store_true")
    ap.add_argument("--timeout", type=float, default=8.0)
    args = ap.parse_args()

    check_platform()
    gpus = check_gpu()
    check_tools()
    check_resources()
    if not args.no_network:
        check_network(args.timeout)

    counts = {}
    for c in CHECKS:
        counts[c["level"]] = counts.get(c["level"], 0) + 1
    if counts.get("FAIL"):
        verdict = "NOT READY: %d 项 FAIL 需要先解决" % counts["FAIL"]
    elif counts.get("WARN"):
        verdict = "基本可用: %d 项 WARN 需要留意" % counts["WARN"]
    else:
        verdict = "READY: 可以按 README 直接安装"
    print("\nMatterGen readiness: " + verdict)
    if args.out:
        with io.open(args.out, "w", encoding="utf-8") as f:
            json.dump({"verdict": verdict, "counts": counts, "gpus": gpus, "checks": CHECKS}, f, indent=2, ensure_ascii=False)
        print("wrote " + args.out)


if __name__ == "__main__":
    main()
