#!/usr/bin/env bash
# MUSE score_extraction 服务器一键环境搭建（离线，2026-08-11）
# 用法: bash setup_server.sh
set -euo pipefail
SE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SE_DIR"

PY_BIN=""
for cand in \
  /home/user/dyy_work/envs/dyy_env/bin/python \
  /home/user/lynsdu2/miniconda3/envs/omnidocbench/bin/python \
  /home/user/lynsdu2/miniconda3/bin/python \
  python3.10 python3; do
  if [ -x "$cand" ] || command -v "$cand" >/dev/null 2>&1; then
    PY_BIN="$cand"
    break
  fi
done
if [ -z "$PY_BIN" ]; then
  echo "错误: 找不到 Python 3.10" >&2
  exit 1
fi
echo "使用解释器: $PY_BIN ($("$PY_BIN" --version))"

# .venv 已预置 wheels/ 与 requirements；无 venv 时创建（不会清空已有内容）
if [ ! -x ".venv/bin/python" ]; then
  "$PY_BIN" -m venv .venv
fi

echo "=== 离线安装依赖 ==="
.venv/bin/python -m pip install --no-index --find-links=.venv/wheels -r .venv/requirements-linux.txt

echo "=== 环境自检 ==="
.venv/bin/python - <<'PY'
import sys
import torch
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "| cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
import numpy, librosa, pretty_midi, mir_eval, music21
print("numpy", numpy.__version__, "| librosa", librosa.__version__, "| music21", music21.__version__)
try:
    import fluidsynth
    print("pyfluidsynth import OK（渲染时还需系统 libfluidsynth）")
except Exception as e:
    print("pyfluidsynth import 失败:", e)
PY

echo "=== 完成 ==="