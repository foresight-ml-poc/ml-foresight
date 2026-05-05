"""Student-owned metrics contract.

Students must implement ``compute_metrics`` to return the evaluation metrics
that matter for their project.
"""

from __future__ import annotations

from typing import Any


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return the metrics used to compare model performance.

    Expected return value:
        A dictionary mapping metric names to numeric values, for example:
        ``{"accuracy": 0.91, "f1": 0.88}``.

    Constraints:
    - Every value must be numeric and convertible to ``float``.
    - Use the same metric set for every model so results remain comparable.
    - Keep metric names stable because they are written to
      ``results/model_metrics.csv``.
    """

    raise NotImplementedError(
        "Implement metrics.compute_metrics() before running scripts/main.py."
    )
