#!/usr/bin/env bash
# Run a teacher under vLLM from the judge venv (.venv-judge) — the same environment as
# scripts/judge/run_judge.sh:
#  - CPATH points at the locally extracted python3.10-dev headers (no sudo on this machine):
#    vLLM's Triton JIT compiles a small C extension that needs Python.h.
#  - VLLM_USE_FLASHINFER_SAMPLER=0: the FlashInfer sampler JIT needs ninja + nvcc at startup;
#    the teacher decodes greedily, so the native torch sampler is fine.
#  - The Hugging Face token is read from ~/.cache/huggingface/token by huggingface_hub.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HDR="$ROOT/.local-pkgs/extracted/usr/include"
if [[ ! -f "$HDR/python3.10/Python.h" ]]; then
  echo "Python headers not found under $HDR — see CLAUDE.md (free-form eval, judge venv setup)" >&2
  exit 1
fi
export CPATH="$HDR:$HDR/python3.10${CPATH:+:$CPATH}"
export C_INCLUDE_PATH="$CPATH"
export PATH="$ROOT/.venv-judge/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
exec "$ROOT/.venv-judge/bin/python" "$ROOT/scripts/distill/generate_teacher_answers.py" "$@"
