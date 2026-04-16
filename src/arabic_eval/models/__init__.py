"""Auto-import model adapters to trigger registry registration.

Imports are guarded so the package can be loaded without torch/transformers
installed (useful for tokenizer-only workflows).
"""
try:
    from arabic_eval.models import llama_adapter  # noqa: F401
except ImportError:
    pass
