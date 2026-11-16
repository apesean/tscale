#!/usr/bin/env bash
# 01_system.sh - 只读的硬件/软件清单采集，不需要 root，不修改任何系统配置。
# 故意不采集 IP / MAC / 序列号 / UUID（结果会提交到公开仓库）。
set -u
export LC_ALL=C

have() { command -v "$1" >/dev/null 2>&1; }
sec()  { printf '\n==================== %s ====================\n' "$1"; }
# run <cmd...>: 打印命令并执行；命令不存在或失败都不会中断脚本
run() {
  if have "$1"; then
    printf '$ %s\n' "$*"
    timeout 30 "$@" 2>&1 | head -n "${MAXLINES:-200}"
  else
    printf '[skip] %s: not found\n' "$1"
  fi
}
show() { # show <file>: 打印一个文件（存在时）
  if [ -r "$1" ]; then printf '%s: %s\n' "$1" "$(tr '\n' ' ' < "$1" | head -c 400)"; fi
}

sec "OS / kernel"
run uname -srvmo
[ -r /etc/os-release ] && grep -E '^(PRETTY_NAME|VERSION_ID|ID)=' /etc/os-release
run uptime
if have systemd-detect-virt; then echo "virt: $(systemd-detect-virt 2>/dev/null || true)"; fi
[ -f /.dockerenv ] && echo "container: docker (/.dockerenv present)"
show /proc/cmdline
echo "glibc: $(getconf GNU_LIBC_VERSION 2>/dev/null || ldd --version 2>&1 | head -1)"

sec "CPU"
MAXLINES=40 run lscpu
echo "logical CPUs usable by this process: $(nproc 2>/dev/null)   installed: $(nproc --all 2>/dev/null)"
echo "--- ISA extensions relevant to vectorized code ---"
flags=$(grep -m1 -E '^(flags|Features)' /proc/cpuinfo 2>/dev/null)
for f in sse4_2 avx avx2 fma avx512f avx512bw avx512vl avx512_vnni avx512_bf16 avx512_fp16 avx_vnni amx_tile amx_bf16 amx_int8 sha_ni vaes neon sve sve2; do
  case " $flags " in *" $f "*) printf '  %-12s yes\n' "$f";; *) printf '  %-12s no\n' "$f";; esac
done
echo "--- frequency scaling ---"
show /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver
show /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
show /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq
show /sys/devices/system/cpu/intel_pstate/no_turbo
show /sys/devices/system/cpu/cpufreq/boost
show /sys/devices/system/cpu/smt/control
echo "--- cache (cpu0) ---"
for d in /sys/devices/system/cpu/cpu0/cache/index*; do
  [ -d "$d" ] && echo "  L$(cat "$d/level") $(cat "$d/type"): $(cat "$d/size") shared_cpu_list=$(cat "$d/shared_cpu_list")"
done

sec "Memory / NUMA"
run free -h
grep -E '^(MemTotal|MemAvailable|SwapTotal|HugePages_Total|HugePages_Free|Hugepagesize|AnonHugePages)' /proc/meminfo
show /sys/kernel/mm/transparent_hugepage/enabled
show /sys/kernel/mm/transparent_hugepage/defrag
show /proc/sys/kernel/numa_balancing
show /proc/sys/vm/swappiness
show /proc/sys/vm/overcommit_memory
if have numactl; then run numactl --hardware; else
  echo "[numactl not found] NUMA nodes from sysfs:"
  for n in /sys/devices/system/node/node*; do
    [ -d "$n" ] && echo "  $(basename "$n"): cpus=$(cat "$n/cpulist") $(grep MemTotal "$n/meminfo" | awk '{print $4/1048576 " GiB"}')"
  done
fi
# DIMM 布局需要 root；有免密 sudo 时才取，并去掉序列号
if have dmidecode && sudo -n true 2>/dev/null; then
  echo "--- DIMMs (dmidecode) ---"
  sudo -n dmidecode -t memory 2>/dev/null | grep -E '^\s+(Size|Type|Speed|Configured Memory Speed|Locator|Manufacturer):' | grep -v 'No Module' | head -80
fi
echo "/dev/shm: $(df -h /dev/shm 2>/dev/null | tail -1)"

sec "Storage"
run lsblk -d -o NAME,SIZE,ROTA,TYPE,TRAN,MODEL
run df -hT -x tmpfs -x devtmpfs -x squashfs -x overlay
echo "--- overlay / root ---"
df -hT / "$HOME" "$PWD" 2>/dev/null | awk '!seen[$0]++'
echo "--- inodes ---"
df -i / "$HOME" 2>/dev/null | awk '!seen[$0]++'
[ -r /proc/mdstat ] && grep -v '^unused' /proc/mdstat | head -20
if have nvme && sudo -n true 2>/dev/null; then sudo -n nvme list 2>&1 | cut -c1-16,39-200 | head -20; fi

sec "GPU / accelerators"
if have lspci; then lspci 2>/dev/null | grep -i -E 'vga|3d|display|accelerat|nvidia|amd.*radeon' || echo "(lspci: no display/accelerator devices)"; else echo "[skip] lspci: not found"; fi
if have nvidia-smi; then
  run nvidia-smi
  echo "--- per-GPU details ---"
  Q=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit,driver_version,vbios_version,compute_mode,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,ecc.mode.current
  nvidia-smi --query-gpu=$Q --format=csv 2>&1 | head -40
  echo "--- compute capability (older drivers do not support this field) ---"
  nvidia-smi --query-gpu=index,name,compute_cap --format=csv 2>&1 | head -20
  echo "--- GPU topology / CPU affinity ---"
  nvidia-smi topo -m 2>&1 | head -40
  echo "--- GPU compute processes right now (count only) ---"
  nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader 2>/dev/null | wc -l
  show /proc/driver/nvidia/version
  echo "persistence / MPS:"; nvidia-smi --query-gpu=index,persistence_mode --format=csv,noheader 2>&1 | head -16
else
  echo "[nvidia-smi not found] -> 没有 NVIDIA 驱动，或者没有 NVIDIA GPU"
  lsmod 2>/dev/null | grep -i -E '^(nvidia|nouveau|amdgpu|i915|xe) ' || true
fi
have rocm-smi && run rocm-smi
have xpu-smi && run xpu-smi discovery
have clinfo && MAXLINES=30 run clinfo -l

sec "CUDA toolkit / GPU kernel dev tools"
for t in nvcc cuda-gdb nsys ncu nvprof; do
  if have $t; then echo "$t: $(command -v $t)"; else echo "$t: not found"; fi
done
have nvcc && nvcc --version 2>&1 | tail -2
echo "CUDA_HOME=${CUDA_HOME:-} CUDA_PATH=${CUDA_PATH:-}"
ls -d /usr/local/cuda* /opt/cuda* 2>/dev/null || echo "(no /usr/local/cuda*)"
echo "--- libs visible to the dynamic linker ---"
ldconfig -p 2>/dev/null | grep -E 'libcuda\.so|libcudart|libcudnn\.so|libnvinfer\.so|libcublas\.so|libnccl|libmkl_rt|libopenblas|libOpenCL' | awk '{print $1, $NF}' | sort -u | head -40
have module && { echo "--- environment modules ---"; module avail 2>&1 | grep -i -E 'cuda|gcc|intel|llvm|mpi|python|conda' | head -30; }

sec "Compilers / build / profiling tools (kernel optimization)"
for t in gcc g++ clang clang++ gfortran icx ld.lld make cmake ninja bazel rustc mpicc; do
  if have $t; then printf '%-10s %s\n' "$t" "$($t --version 2>&1 | head -1)"; else printf '%-10s not found\n' "$t"; fi
done
if have go; then printf '%-10s %s\n' go "$(go version 2>&1 | head -1)"; else printf '%-10s not found\n' go; fi
have gcc && echo "gcc -march=native resolves to: $(gcc -march=native -Q --help=target 2>/dev/null | awk '/-march=/{print $2; exit}')"
have gcc && { echo 'int main(){return 0;}' | gcc -fopenmp -x c - -o /dev/null 2>/dev/null && echo "OpenMP (gcc -fopenmp): ok" || echo "OpenMP (gcc -fopenmp): FAILED"; }
echo "--- tracing / profiling ---"
for t in perf bpftrace bpftool trace-cmd strace ltrace gdb valgrind numactl hwloc-ls turbostat cpupower tuned-adm fio stress-ng; do
  if have $t; then printf '%-10s %s\n' "$t" "$(command -v $t)"; else printf '%-10s not found\n' "$t"; fi
done
show /proc/sys/kernel/perf_event_paranoid
show /proc/sys/kernel/kptr_restrict
show /proc/sys/kernel/unprivileged_bpf_disabled
show /proc/sys/kernel/sched_autogroup_enabled
echo "--- Linux kernel build prerequisites ---"
kr=$(uname -r)
[ -d "/lib/modules/$kr/build" ] && echo "kernel headers: /lib/modules/$kr/build present" || echo "kernel headers: NOT installed for $kr"
if [ -r "/boot/config-$kr" ]; then echo "kernel config: /boot/config-$kr"; cfg="/boot/config-$kr"
elif [ -r /proc/config.gz ]; then echo "kernel config: /proc/config.gz"; cfg=/proc/config.gz
else echo "kernel config: not readable"; cfg=""; fi
if [ -n "$cfg" ]; then
  (case "$cfg" in *.gz) zcat "$cfg";; *) cat "$cfg";; esac) 2>/dev/null | grep -E '^CONFIG_(PREEMPT[A-Z_]*|HZ|NO_HZ_FULL|NUMA_BALANCING|TRANSPARENT_HUGEPAGE|BPF_SYSCALL|DEBUG_INFO_BTF|FTRACE|KPROBES|SCHED_CLASS_EXT|CGROUP_SCHED)=' | head -30
fi
[ -r /sys/kernel/btf/vmlinux ] && echo "BTF: /sys/kernel/btf/vmlinux present"
[ -d /sys/firmware/efi ] && echo "boot: UEFI" || echo "boot: legacy BIOS (or container)"
have mokutil && mokutil --sb-state 2>&1 | head -1
echo "--- CPU vulnerability mitigations ---"
grep -H . /sys/devices/system/cpu/vulnerabilities/* 2>/dev/null | sed 's#.*/##' | head -30

sec "Privileges / limits"
echo "uid=$(id -u) (root=$([ "$(id -u)" = 0 ] && echo yes || echo no))"
if have sudo; then sudo -n true 2>/dev/null && echo "sudo: passwordless OK" || echo "sudo: installed, needs password or not permitted"; else echo "sudo: not found"; fi
id -nG 2>/dev/null | tr ' ' '\n' | grep -E '^(docker|sudo|wheel|video|render|kvm)$' | tr '\n' ' '; echo
echo "ulimit: nofile=$(ulimit -n) nproc=$(ulimit -u) memlock=$(ulimit -l) stack=$(ulimit -s)"
if [ -r /sys/fs/cgroup/cpu.max ]; then show /sys/fs/cgroup/cpu.max; show /sys/fs/cgroup/memory.max; fi
have sinfo && { echo "--- slurm ---"; sinfo -o '%P %a %l %D %c %m %G' 2>&1 | head -20; }
have qstat && echo "PBS/SGE qstat present"

sec "Python / package managers / containers"
for t in python3 python3.10 python3.11 python3.12 python pip3 uv conda mamba micromamba pipx; do
  if have $t; then printf '%-12s %s  (%s)\n' "$t" "$($t --version 2>&1 | head -1)" "$(command -v $t)"; else printf '%-12s not found\n' "$t"; fi
done
have conda && { echo "--- conda envs ---"; conda env list 2>&1 | grep -v '^#' | head -20; }
for t in git git-lfs docker podman apptainer singularity nvidia-container-cli nvidia-ctk tmux screen wget curl; do
  if have $t; then printf '%-22s %s\n' "$t" "$($t --version 2>&1 | head -1)"; else printf '%-22s not found\n' "$t"; fi
done
have docker && { docker info 2>/dev/null | grep -i -E 'runtimes|default runtime|server version' || echo "docker: daemon not reachable for this user"; }
echo "proxy env: http_proxy=$([ -n "${http_proxy:-${HTTP_PROXY:-}}" ] && echo set || echo unset) https_proxy=$([ -n "${https_proxy:-${HTTPS_PROXY:-}}" ] && echo set || echo unset)"

sec "Network hardware (no addresses)"
for n in /sys/class/net/*; do
  i=$(basename "$n"); [ "$i" = lo ] && continue
  echo "  $i: state=$(cat "$n/operstate" 2>/dev/null) speed=$(cat "$n/speed" 2>/dev/null || echo ?)Mb/s mtu=$(cat "$n/mtu" 2>/dev/null)"
done
have ibstat && ibstat 2>/dev/null | grep -E 'CA type|State|Rate' | head -12

sec "Sensors"
if have sensors; then MAXLINES=40 run sensors; else echo "[skip] sensors: not found"; fi
echo
echo "[01_system.sh done]"
