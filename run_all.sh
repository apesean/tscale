#!/usr/bin/env bash
# run_all.sh - 一键采集硬件信息 + 跑基准，结果写到 results/<TAG>/report.md
#
#   ./run_all.sh                      # 默认：全部跑一遍（无 GPU/torch 的部分会自动跳过）
#   QUICK=1 ./run_all.sh              # 快速模式（约 1 分钟，数字不够准）
#   BENCH_DIRS="/data /scratch" ./run_all.sh     # 额外测这些目录所在的磁盘
#   TORCH_PYTHON=/path/to/env/bin/python ./run_all.sh   # 用已有的、装了 torch 的解释器跑 GPU 基准
#   SETUP_TORCH=1 ./run_all.sh        # 没有 torch 时，在 .venv-probe 里 pip install torch（约 2-3 GB）
#   FORCE_GPU=1 ./run_all.sh          # GPU 上已有任务时也照样测
#   NO_NETWORK=1 ./run_all.sh         # 不做外网连通性探测
#   NO_REDACT=1 ./run_all.sh          # 不替换报告里的主机名/用户名
#   TAG=myserver ./run_all.sh         # 自定义结果目录名
#
# 不需要 root，不修改系统配置。磁盘测试只写一个 1 GiB 的临时文件并在结束后删除。
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8   # 老版本 Python 在 C locale 下打印中文会报错

TAG=${TAG:-server-$(date +%Y%m%d-%H%M%S)}
OUT=results/$TAG
mkdir -p "$OUT"
QUICK_FLAG=""; [ -n "${QUICK:-}" ] && QUICK_FLAG="--quick"

PY=${PYTHON:-$(command -v python3 || true)}
step() { printf '\n\033[1m>>> %s\033[0m\n' "$1"; }

step "1/4 system inventory -> $OUT/system.txt"
bash probe/01_system.sh > "$OUT/system.txt" 2>&1
echo "    $(wc -l < "$OUT/system.txt") lines"

if [ -z "$PY" ]; then
  echo "python3 not found: skipping benchmarks and readiness check" | tee "$OUT/cpu.txt"
else
  step "2/4 CPU / memory / disk benchmark -> $OUT/cpu.txt"
  # shellcheck disable=SC2086
  "$PY" probe/02_bench_cpu.py --out "$OUT/cpu.json" $QUICK_FLAG --dirs "$PWD" ${BENCH_DIRS:-} 2>&1 | tee "$OUT/cpu.txt"

  step "3/4 GPU benchmark -> $OUT/gpu.txt"
  has_torch() { [ -x "$1" ] && "$1" -c 'import torch' >/dev/null 2>&1; }
  TPY=""
  for cand in "${TORCH_PYTHON:-}" "$PY" ".venv-probe/bin/python"; do
    if [ -n "$cand" ] && has_torch "$cand"; then TPY=$cand; break; fi
  done
  if [ -z "$TPY" ] && [ -n "${SETUP_TORCH:-}" ]; then
    echo "    creating .venv-probe and installing torch (this downloads ~2-3 GB)..."
    "$PY" -m venv .venv-probe && .venv-probe/bin/python -m pip install -q --upgrade pip \
      && .venv-probe/bin/python -m pip install -q torch 2>&1 | tail -5
    has_torch .venv-probe/bin/python && TPY=.venv-probe/bin/python
  fi
  if [ -n "$TPY" ]; then
    FORCE=""; [ -n "${FORCE_GPU:-}" ] && FORCE="--force"
    # shellcheck disable=SC2086
    "$TPY" probe/03_bench_gpu.py --out "$OUT/gpu.json" $QUICK_FLAG $FORCE 2>&1 | tee "$OUT/gpu.txt"
  else
    {
      echo "SKIPPED: 没有找到装了 PyTorch 的 Python。"
      echo "  方式 a) TORCH_PYTHON=/path/to/env/bin/python ./run_all.sh"
      echo "  方式 b) SETUP_TORCH=1 ./run_all.sh   (在 .venv-probe 里装 torch，约 2-3 GB)"
      echo "GPU 的型号/显存/驱动信息仍然在 system.txt 里。"
    } | tee "$OUT/gpu.txt"
  fi

  step "4/4 MatterGen readiness -> $OUT/mattergen.txt"
  NET=""; [ -n "${NO_NETWORK:-}" ] && NET="--no-network"
  # shellcheck disable=SC2086
  "$PY" probe/04_mattergen_ready.py --out "$OUT/mattergen.json" $NET 2>&1 | tee "$OUT/mattergen.txt"
fi

# ---- 合并成一个 report.md，方便直接贴回对话 ----
{
  echo "# tscale hardware probe: $TAG"
  echo
  echo "Generated $(date -u +%Y-%m-%dT%H:%MZ) by run_all.sh${QUICK:+ (QUICK mode)}"
  for part in mattergen gpu cpu system; do
    [ -f "$OUT/$part.txt" ] || continue
    printf '\n## %s\n\n```text\n' "$part"
    cat "$OUT/$part.txt"
    printf '```\n'
  done
} > "$OUT/report.md"

# ---- 脱敏：结果会进公开仓库，把主机名和用户名换成占位符 ----
if [ -z "${NO_REDACT:-}" ]; then
  H=$(hostname 2>/dev/null || true); HS=${H%%.*}; U=$(id -un 2>/dev/null || true)
  for f in "$OUT"/*; do
    [ -f "$f" ] || continue
    [ ${#H} -ge 3 ] && sed -i "s/$(printf '%s' "$H" | sed 's/[][\.*^$/]/\\&/g')/<host>/g" "$f"
    [ ${#HS} -ge 3 ] && sed -i "s/$(printf '%s' "$HS" | sed 's/[][\.*^$/]/\\&/g')/<host>/g" "$f"
    if [ ${#U} -ge 3 ] && [ "$U" != root ]; then
      sed -i "s/\b$(printf '%s' "$U" | sed 's/[][\.*^$/]/\\&/g')\b/<user>/g" "$f"
    fi
  done
fi

step "done"
cat <<EOF
结果目录: $OUT
  report.md      合并后的完整报告（把这个文件发回给 Claude 即可）
  *.json         机器可读的基准数字

把结果传回 GitHub（提交前可以先看一眼 report.md 里有没有不想公开的内容）:
  git add $OUT && git commit -m "probe results: $TAG" && git push

如果这台机器推不了 GitHub，直接把 $OUT/report.md 的内容贴到对话里也行。
EOF
