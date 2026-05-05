"""Auto-import task modules to trigger registry registration.

Imports are guarded so the package can be loaded without torch/transformers
installed (useful for tokenizer-only workflows).
"""
try:
    from arabic_eval.tasks import lighteval  # noqa: F401
except ImportError:
    pass
