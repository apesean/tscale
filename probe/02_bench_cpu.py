#!/usr/bin/env python3
"""02_bench_cpu.py - CPU / 内存带宽 / 磁盘 的快速基准。

只依赖 Python 标准库 (>=3.6)。有 gcc 时会额外编译 stream_mini.c 测内存带宽，
有 numpy 时会测 BLAS。磁盘测试只在指定目录里写一个临时文件，结束后删除。

用法:
    python3 02_bench_cpu.py [--out cpu.json] [--dirs DIR [DIR ...]] [--disk-mb 1024] [--quick]
"""
from __future__ import print_function

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))


def usable_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# ---------------------------------------------------------------- CPU
def _py_loop(n):
    s = 0
    for i in range(n):
        s += i * i % 7
    return s


def bench_python_single(quick):
    n = 2000000 if quick else 10000000
    t = time.perf_counter()
    _py_loop(n)
    dt = time.perf_counter() - t
    return {"iters": n, "seconds": round(dt, 3), "Mops_per_s": round(n / dt / 1e6, 2)}


def _zlib_worker(seconds):
    # 1 MiB 的半可压缩数据；zlib 是通用的整数/分支型负载
    block = (os.urandom(4096) + b"\0" * 4096) * 128
    n = 0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        zlib.compress(block, 6)
        n += 1
    return n


def bench_scaling(quick):
    ncpu = usable_cpus()
    seconds = 2.0 if quick else 4.0
    counts = sorted(set([1, max(1, ncpu // 4), max(1, ncpu // 2), ncpu]))
    out = []
    base = None
    for k in counts:
        pool = mp.Pool(k)
        try:
            done = sum(pool.map(_zlib_worker, [seconds] * k))
        finally:
            pool.close()
            pool.join()
        rate = done / seconds  # MiB/s compressed
        if base is None:
            base = rate
        out.append({
            "procs": k,
            "MiB_per_s": round(rate, 1),
            "speedup": round(rate / base, 2),
            "efficiency": round(rate / base / k, 2),
        })
        print("  zlib x%-4d %8.1f MiB/s  speedup %.2f  eff %.0f%%" % (k, rate, rate / base, 100 * rate / base / k))
    return out


# ---------------------------------------------------------------- memory
def bench_memcpy_python(quick):
    size = (256 if quick else 1024) * 1024 * 1024
    try:
        src = bytearray(size)
    except MemoryError:
        return {"error": "MemoryError allocating %d MiB" % (size >> 20)}
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        dst = bytes(src)
        dt = time.perf_counter() - t
        del dst
        best = min(best, dt)
    return {"size_MiB": size >> 20, "single_thread_copy_GBps": round(size / best / 1e9, 2)}


def bench_stream(quick):
    cc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if not cc:
        return {"skipped": "no C compiler"}
    tmp = tempfile.mkdtemp(prefix="tscale_stream_")
    exe = os.path.join(tmp, "stream_mini")
    src = os.path.join(HERE, "stream_mini.c")
    res = {"compiler": cc}
    try:
        built = False
        for flags in (["-O3", "-march=native", "-fopenmp"], ["-O3", "-fopenmp"], ["-O3"]):
            p = subprocess.Popen([cc] + flags + [src, "-o", exe], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, universal_newlines=True)
            log = p.communicate()[0]
            if p.returncode == 0:
                res["flags"] = " ".join(flags)
                built = True
                break
        if not built:
            res["error"] = "compile failed: " + log[-400:]
            return res
        mb = "128" if quick else "320"
        ncpu = usable_cpus()
        runs = []
        for threads in sorted(set([1, ncpu])):
            env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_PROC_BIND="spread", OMP_PLACES="cores")
            p = subprocess.Popen([exe, mb], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 universal_newlines=True, env=env)
            txt = p.communicate()[0]
            if p.returncode != 0:
                runs.append({"threads": threads, "error": txt[-300:]})
                continue
            kv = dict(line.split("=", 1) for line in txt.strip().splitlines() if "=" in line)
            r = {"threads": int(kv.get("threads", threads))}
            for key in ("copy_gbps", "triad_gbps", "incache_gflops"):
                if key in kv:
                    r[key] = float(kv[key])
            runs.append(r)
            print("  stream x%-4d copy %6.1f GB/s  triad %6.1f GB/s  in-cache FP %7.1f GFLOP/s" % (
                r["threads"], r.get("copy_gbps", 0), r.get("triad_gbps", 0), r.get("incache_gflops", 0)))
        res["runs"] = runs
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def bench_numpy(quick):
    try:
        import numpy as np
    except Exception as e:  # noqa
        return {"skipped": "numpy not importable (%s)" % type(e).__name__}
    n = 2048 if quick else 4096
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    a.dot(b)
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        a.dot(b)
        best = min(best, time.perf_counter() - t)
    res = {"numpy": np.__version__, "n": n, "dgemm_GFLOPS": round(2.0 * n ** 3 / best / 1e9, 1)}
    try:
        from threadpoolctl import threadpool_info
        res["blas"] = [{k: d.get(k) for k in ("internal_api", "num_threads", "architecture")} for d in threadpool_info()]
    except Exception:
        pass
    print("  numpy dgemm n=%d: %.1f GFLOP/s" % (n, res["dgemm_GFLOPS"]))
    return res


# ---------------------------------------------------------------- disk
def bench_disk(path, size_mb, quick):
    res = {"dir": path}
    if not os.path.isdir(path) or not os.access(path, os.W_OK):
        res["skipped"] = "not a writable directory"
        return res
    st = os.statvfs(path)
    free_mb = st.f_bavail * st.f_frsize // (1024 * 1024)
    res["free_GiB"] = round(free_mb / 1024.0, 1)
    if free_mb < size_mb * 4:
        res["skipped"] = "less than %d MiB free, not touching it" % (size_mb * 4)
        return res
    d = tempfile.mkdtemp(prefix=".tscale_disk_", dir=path)
    try:
        fn = os.path.join(d, "seq.bin")
        chunk = os.urandom(4 * 1024 * 1024)
        nchunks = size_mb // 4
        # sequential write, fsync included in the timing
        t = time.perf_counter()
        fd = os.open(fn, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        for _ in range(nchunks):
            os.write(fd, chunk)
        os.fsync(fd)
        os.close(fd)
        dt = time.perf_counter() - t
        res["seq_write_MBps"] = round(nchunks * 4 / dt, 1)
        # drop this file from the page cache (no root needed) and read it back
        fd = os.open(fn, os.O_RDONLY)
        cold = False
        if hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                cold = True
            except OSError:
                pass
        t = time.perf_counter()
        while os.read(fd, 4 * 1024 * 1024):
            pass
        dt = time.perf_counter() - t
        res["seq_read_MBps"] = round(nchunks * 4 / dt, 1)
        res["seq_read_cache_dropped"] = cold
        # 4 KiB random reads on the (again cold) file
        if cold:
            import random
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            nread = 2000 if quick else 8000
            pages = size_mb * 256
            t = time.perf_counter()
            for _ in range(nread):
                os.pread(fd, 4096, random.randrange(pages) * 4096)
            dt = time.perf_counter() - t
            res["rand_read_4k_IOPS"] = int(nread / dt)
        os.close(fd)
        os.unlink(fn)
        # many small files: proxy for dataset caches / pip installs
        nfiles = 500 if quick else 2000
        small = os.urandom(8192)
        sub = os.path.join(d, "small")
        os.mkdir(sub)
        t = time.perf_counter()
        for i in range(nfiles):
            with open(os.path.join(sub, "%05d" % i), "wb") as f:
                f.write(small)
        fd = os.open(sub, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
        dt = time.perf_counter() - t
        res["small_file_create_per_s"] = int(nfiles / dt)
    except OSError as e:
        res["error"] = "%s: %s" % (type(e).__name__, e)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  disk %s: write %s MB/s, read %s MB/s, 4k rand read %s IOPS, small files %s/s" % (
        path, res.get("seq_write_MBps", "?"), res.get("seq_read_MBps", "?"),
        res.get("rand_read_4k_IOPS", "?"), res.get("small_file_create_per_s", "?")))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--dirs", nargs="*", default=None, help="directories to disk-benchmark (default: cwd)")
    ap.add_argument("--disk-mb", type=int, default=1024)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-disk", action="store_true")
    args = ap.parse_args()

    result = {"python": sys.version.split()[0], "usable_cpus": usable_cpus(), "cpu_count": os.cpu_count()}
    print("usable CPUs: %d (of %s)" % (result["usable_cpus"], result["cpu_count"]))

    print("[1/5] single-thread Python loop")
    result["python_single"] = bench_python_single(args.quick)
    print("  %.2f M iter/s" % result["python_single"]["Mops_per_s"])

    print("[2/5] multi-process scaling (zlib)")
    result["scaling_zlib"] = bench_scaling(args.quick)

    print("[3/5] memory bandwidth")
    result["stream"] = bench_stream(args.quick)
    if "runs" not in result["stream"]:
        print("  stream_mini unavailable (%s); falling back to Python memcpy" %
              (result["stream"].get("skipped") or result["stream"].get("error")))
        result["memcpy_python"] = bench_memcpy_python(args.quick)
        print("  %s" % result["memcpy_python"])

    print("[4/5] numpy / BLAS")
    result["numpy"] = bench_numpy(args.quick)
    if "skipped" in result["numpy"]:
        print("  " + result["numpy"]["skipped"])

    print("[5/5] disk")
    result["disk"] = []
    if not args.no_disk:
        dirs = args.dirs or [os.getcwd()]
        size = 256 if args.quick else args.disk_mb
        for d in dirs:
            result["disk"].append(bench_disk(os.path.abspath(d), size, args.quick))
            if "skipped" in result["disk"][-1]:
                print("  disk %s: skipped (%s)" % (d, result["disk"][-1]["skipped"]))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print("wrote " + args.out)


if __name__ == "__main__":
    main()
