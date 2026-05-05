"""Fixed Streamlit entry point for the project template."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import MODEL_METRICS_FILE


def build_app() -> None:
    """Render the project Streamlit application.

    Students should replace the placeholder sections with their own visualizations,
    explanations, and prediction workflow. The function name and file location are
    fixed because ``scripts/main.py`` launches Streamlit with this module.
    """

    st.set_page_config(page_title="ML Project Template", layout="wide")

    st.title("Machine Learning Proof of Concept")
    st.write(
        "Update `src/app.py` to present your business objective, data insights, "
        "model comparison, and final demo."
    )

    st.subheader("Expected student customizations")
    st.markdown(
        """
        - Describe the business objective and dataset.
        - Show relevant plots and key findings.
        - Explain the selected models and their trade-offs.
        - Add widgets or predictions if your project needs an interactive demo.
        """
    )

    st.subheader("Latest evaluation results")
    if MODEL_METRICS_FILE.exists():
        metrics_df = pd.read_csv(MODEL_METRICS_FILE)
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.info(
            "Run `python scripts/main.py` after training your models to generate "
            "`results/model_metrics.csv`."
        )


if __name__ == "__main__":
    build_app()
