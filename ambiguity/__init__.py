"""Clarion ambiguity package.

Owns: dataset generation (``generate_data``), the schema-aware 3-class
classifier (``classifier``), debugging/eval entry (``self_eval``), and the
trained artifacts under ``ambiguity/artifacts/``.

``__init__`` is deliberately light (no torch import at package import time):
:class:N = ``AmbiguityClassifier`` is lazily exposed via module ``__getattr__``
so ``import ambiguity.analysis`` stays cheap and safe.
"""
