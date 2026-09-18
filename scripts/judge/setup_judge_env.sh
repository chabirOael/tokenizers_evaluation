#!/usr/bin/env bash
# Reproduce the free-form judge environment, without sudo:
#   .venv-judge          vLLM (CUDA 12.9 wheel — the driver on this machine is CUDA 12.8 and
#                        PyPI's default vLLM pulls a CUDA 13 torch that cannot initialize on it)
#                        + the matching torch from the PyTorch cu129 index, pyarrow, ninja
#   .local-pkgs/         python3.X-dev headers extracted from the Ubuntu packages (Triton's JIT
#                        compiles a C extension that needs Python.h; apt-get download + dpkg -x
#                        need no root). scripts/judge/run_judge.sh puts them on CPATH.
# Verified 2026-09-18 with vLLM 0.29.0 / torch 2.13.0+cu129 on an H100, driver 570.211.
#
#   scripts/judge/setup_judge_env.sh                 # default versions below
#   VLLM_VERSION=0.29.0 CUDA_TAG=cu129 scripts/judge/setup_judge_env.sh
#
# Afterwards download the judge weights once (public repo, ~62 GB):
#   .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-31b-it')"
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
CUDA_TAG="${CUDA_TAG:-cu129}"
PYTHON="${PYTHON:-python3}"
VENV="$ROOT/.venv-judge"
WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${CUDA_TAG}-cp38-abi3-manylinux_2_28_x86_64.whl"

echo "== venv: $VENV"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
echo "== installing vllm ${VLLM_VERSION}+${CUDA_TAG} (torch from the ${CUDA_TAG} index)"
"$VENV/bin/pip" install -q "vllm @ ${WHEEL}" --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
"$VENV/bin/pip" install -q pyarrow ninja

PYVER="$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
HDR="$ROOT/.local-pkgs/extracted/usr/include/python${PYVER}/Python.h"
if [[ ! -f "$HDR" ]]; then
  echo "== extracting python${PYVER}-dev headers under .local-pkgs/extracted (no root needed)"
  TMP="$(mktemp -d)"
  ( cd "$TMP" && apt-get download "python${PYVER}-dev" "libpython${PYVER}-dev" )
  mkdir -p "$ROOT/.local-pkgs/extracted"
  for d in "$TMP"/*.deb; do dpkg -x "$d" "$ROOT/.local-pkgs/extracted"; done
  rm -rf "$TMP"
fi

echo "== verifying"
export CPATH="$ROOT/.local-pkgs/extracted/usr/include:$ROOT/.local-pkgs/extracted/usr/include/python${PYVER}"
printf '#include <Python.h>\nint main(){return 0;}\n' > /tmp/_hdr_check_$$.c
gcc -c /tmp/_hdr_check_$$.c -o /tmp/_hdr_check_$$.o && rm -f /tmp/_hdr_check_$$.c /tmp/_hdr_check_$$.o
"$VENV/bin/python" - <<'PY'
import torch, vllm
from vllm.model_executor.models.registry import _VLLM_MODELS as M
print(f"vllm {vllm.__version__} | torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "torch cannot see the GPU — wrong CUDA_TAG for this driver?"
assert any("Gemma4" in k for k in M), "this vLLM has no Gemma 4 support"
print("Gemma 4 architecture registered; headers compile. Judge env OK.")
PY
