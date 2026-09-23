"""Sequence-level distillation from a third-party Arabic teacher (2026-09-23).

``teacher`` is stdlib + PyYAML only (it is imported by the vLLM generation
script in ``.venv-judge``); ``postprocess`` applies the free-form eval's text
rules to teacher answers and needs the main venv.
"""
