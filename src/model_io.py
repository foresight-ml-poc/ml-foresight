"""Helpers for loading serialized models."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def load_model(model_path: Path) -> Any:
    """Load a serialized model from disk.

    Supported formats are `.joblib`, `.pkl`, and `.pickle`.
    """

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    suffix = model_path.suffix.lower()

    if suffix == ".joblib":
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                "Loading `.joblib` files requires the `joblib` package. "
                "Add it to requirements.txt if needed."
            ) from exc

        return joblib.load(model_path)

    if suffix in {".pkl", ".pickle"}:
        with model_path.open("rb") as file_handle:
            return pickle.load(file_handle)

    raise ValueError(
        f"Unsupported model format for {model_path}. Use .joblib, .pkl, or .pickle."
    )
