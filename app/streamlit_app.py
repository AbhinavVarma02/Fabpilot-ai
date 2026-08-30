"""Streamlit dashboard for FabPilot AI: an engineer-facing yield diagnosis copilot.

This file is presentation and orchestration only. All predictions come from the
trained model artifacts routed by the LangGraph workflow; the UI just renders the
structured evidence. Model, agent, SHAP, and preprocessing logic live in ``src/``
and are not touched here.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import ui
from src.agent.graph import build_fabpilot_graph
from src.data.demo_data import load_dashboard_secom_features, wafer_demo_sample_path


st.set_page_config(page_title="FabPilot AI · Yield Diagnosis Copilot", page_icon="🔬", layout="wide")


@st.cache_data(show_spinner=False)
def load_secom_cached() -> pd.DataFrame:
    return load_dashboard_secom_features()


@st.cache_data(show_spinner=False)
def load_wafer_sample():
    path = wafer_demo_sample_path()
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["class_names"].astype(str).tolist()


@st.cache_resource(show_spinner=False)
def load_graph():
    return build_fabpilot_graph()


def required_artifacts_exist() -> list[str]:
    required = [
        PROJECT_ROOT / "artifacts" / "secom_yield_model.joblib",
        PROJECT_ROOT / "artifacts" / "wafer_cnn.pt",
        PROJECT_ROOT / "data" / "demo" / "secom_demo_features.csv",
        PROJECT_ROOT / "data" / "demo" / "wm811k_demo_sample.npz",
    ]
    return [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]


ui.inject_css()
ui.hero()

missing = required_artifacts_exist()
if missing:
    ui.missing_artifacts(missing)
    st.stop()

X_secom = load_secom_cached()
X_wafers, y_wafers, class_names = load_wafer_sample()
graph = load_graph()


def wafer_label(index: int) -> str:
    return class_names[int(y_wafers[int(index)])]


# --- Sidebar brand/about panel ----------------------------------------------
with st.sidebar:
    ui.sidebar_brand()
    st.divider()
    ui.sidebar_heading("About this demo")
    ui.sidebar_footer()


# --- Investigation setup (main page) ----------------------------------------
with st.container(border=True):
    st.markdown("**Investigation setup**")
    secom_col, wafer_col, run_col = st.columns([0.36, 0.36, 0.28], vertical_alignment="bottom")
    with secom_col:
        secom_index = st.number_input(
            "SECOM sensor row",
            min_value=0,
            max_value=len(X_secom) - 1,
            value=0,
            step=1,
            help="Index of the anonymized process-sensor row to score.",
        )
    with wafer_col:
        wafer_index = st.number_input(
            "Wafer-map sample",
            min_value=0,
            max_value=len(X_wafers) - 1,
            value=0,
            step=1,
            help="Index of the wafer map to classify. Ground-truth label shown in the preview.",
        )
    with run_col:
        run = st.button("▶  Run investigation", type="primary", width="stretch")

# Inference runs only on an explicit button click, so a page refresh never triggers
# model work or any paid API usage. Results persist between reruns via session_state.
if run:
    with st.spinner("Routing sensor row and wafer map through the workflow…"):
        state = graph.invoke(
            {
                "sensor_data": X_secom.iloc[[int(secom_index)]],
                "wafer_image": X_wafers[int(wafer_index), 0],
            }
        )
    st.session_state["result"] = state
    st.session_state["result_index"] = (int(secom_index), int(wafer_index))

result = st.session_state.get("result")

if result is None:
    # Empty state: workflow explainer beside a live preview of the selected wafer.
    intro_col, preview_col = st.columns([0.62, 0.38])
    with intro_col:
        ui.empty_state()
    with preview_col:
        with st.container(border=True):
            st.markdown("**Selected wafer preview**")
            fig = ui.wafer_figure(X_wafers[int(wafer_index), 0])
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            st.caption(f"Ground-truth label: {wafer_label(wafer_index)}")
else:
    secom_idx, wafer_idx = st.session_state.get("result_index", (int(secom_index), int(wafer_index)))

    ui.status_banner(result, secom_idx, wafer_idx)
    ui.metric_cards(result)

    wafer_col, shap_col = st.columns([0.42, 0.58])

    with wafer_col:
        with st.container(border=True):
            st.markdown("**Wafer defect pattern**")
            fig = ui.wafer_figure(X_wafers[int(wafer_idx), 0])
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            defect_class = result.get("defect_class", "n/a")
            defect_conf = float(result.get("defect_confidence") or 0.0)
            st.caption(
                f"Predicted: **{defect_class}** | softmax confidence {ui.format_probability(defect_conf)} | "
                f"ground truth {wafer_label(wafer_idx)}"
            )

    with shap_col:
        with st.container(border=True):
            st.markdown("**Top sensor-feature contributions (SHAP)**")
            shap_features = result.get("shap_features") or []
            if shap_features:
                fig = ui.shap_figure(shap_features)
                st.pyplot(fig, width="stretch")
                plt.close(fig)
                ui.shap_table(shap_features)
            else:
                st.info("No SHAP contributions were produced for this sample.")

    st.markdown("#### Structured investigation evidence")
    ui.evidence_sections(result)
    ui.human_review(result)

    with st.expander("View / download the full text report"):
        summary = result.get("final_summary") or "No summary generated."
        st.text(summary)
        st.download_button(
            "Download report (.txt)",
            summary,
            file_name=f"fabpilot_investigation_secom{secom_idx}_wafer{wafer_idx}.txt",
            mime="text/plain",
        )

    ui.safety_note()
