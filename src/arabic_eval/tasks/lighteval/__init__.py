"""LightEval-based Arabic MCQ benchmarks.

Importing this package auto-imports every benchmark module, which fires each
class's ``@task_registry.register(...)`` decorator. Adding a new benchmark
means dropping a new module next to the existing four and adding it to the
import line below — nothing else.
"""
from arabic_eval.tasks.lighteval import (  # noqa: F401
    acva,
    alghafa,
    arabic_exam,
    culture_arabic_mmlu,
)
from arabic_eval.tasks.lighteval.base import LightEvalBenchmarkTask  # noqa: F401
