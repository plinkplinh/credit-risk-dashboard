
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Credit Risk PoC",
    page_icon="📊",
    layout="wide",
)


# =====================================================
# FILE PATHS
# =====================================================

APP_DIR = Path(__file__).resolve().parent

MODEL_DIR = APP_DIR / "artifacts" / "model"
SURVEY_DIR = APP_DIR / "artifacts" / "survey"

MODEL_PATH = (
    MODEL_DIR /
    "home_credit_benchmark_model.joblib"
)

METRICS_PATH = (
    MODEL_DIR /
    "test_model_metrics.csv"
)

POC_SCORES_PATH = (
    MODEL_DIR /
    "poc_scenario_scores.csv"
)

POC_SUMMARY_PATH = (
    MODEL_DIR /
    "poc_scenario_summary.csv"
)

REPRESENTATIVE_CASES_PATH = (
    MODEL_DIR /
    "poc_three_representative_cases.csv"
)

SAMPLE_FLOW_PATH = (
    SURVEY_DIR /
    "sample_flow.csv"
)


# =====================================================
# LOAD FUNCTIONS
# =====================================================

@st.cache_resource
def load_model_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


# =====================================================
# LOAD AND VALIDATE
# =====================================================

try:
    model_artifact = load_model_artifact()

    model_metrics = load_csv(METRICS_PATH)
    poc_scores = load_csv(POC_SCORES_PATH)
    poc_summary = load_csv(POC_SUMMARY_PATH)

    representative_cases = load_csv(
        REPRESENTATIVE_CASES_PATH
    )

    sample_flow = load_csv(SAMPLE_FLOW_PATH)

except Exception as error:
    st.error(
        "The application could not load its "
        "research artifacts."
    )
    st.exception(error)
    st.stop()


required_model_keys = {
    "model_pipeline",
    "platt_calibrator",
    "decision_threshold",
    "raw_predictor_columns",
    "claim_boundary",
}

missing_model_keys = (
    required_model_keys
    .difference(model_artifact.keys())
)

if missing_model_keys:
    st.error(
        "The model bundle is missing: "
        + ", ".join(sorted(missing_model_keys))
    )
    st.stop()


expected_shapes = {
    "PoC scores": (len(poc_scores), 900),
    "PoC summary": (len(poc_summary), 3),
    "Representative cases": (
        len(representative_cases),
        3,
    ),
}

invalid_shapes = [
    f"{name}: observed {observed}, expected {expected}"
    for name, (observed, expected)
    in expected_shapes.items()
    if observed != expected
]

if invalid_shapes:
    st.error(
        "Unexpected PoC file sizes:\n"
        + "\n".join(invalid_shapes)
    )
    st.stop()


# =====================================================
# MINIMAL TEST INTERFACE
# =====================================================

st.title(
    "Alternative-Data Credit Risk Assessment"
)

st.subheader(
    "Proof-of-Concept Artifact Test"
)

st.warning(
    "Research demonstration only. This is a "
    "technical Home Credit benchmark and not a "
    "validated Vietnamese lending system."
)

st.success(
    "All core model and survey artifacts loaded "
    "successfully."
)


column1, column2, column3, column4 = st.columns(4)

column1.metric(
    "Model predictors",
    len(model_artifact["raw_predictor_columns"]),
)

column2.metric(
    "PoC observations",
    len(poc_scores),
)

column3.metric(
    "Scenario cohorts",
    len(poc_summary),
)

column4.metric(
    "Representative cases",
    len(representative_cases),
)


st.subheader("Model bundle")

st.write(
    "Illustrative analytical threshold:",
    f"{model_artifact['decision_threshold']:.4f}",
)

st.write(
    "Claim boundary:",
    model_artifact["claim_boundary"],
)


st.subheader("PoC scenario summary")

display_summary = poc_summary.copy()

percentage_columns = [
    "mean",
    "median",
    "minimum",
    "maximum",
    "q05",
    "q25",
    "q75",
    "q95",
    "proportion_above_illustrative_threshold",
]

for column in percentage_columns:
    if column in display_summary.columns:
        display_summary[column] = (
            display_summary[column]
            .map(lambda value: f"{value:.2%}")
        )

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True,
)


st.subheader("Survey sample flow")

st.dataframe(
    sample_flow,
    use_container_width=True,
    hide_index=True,
)


st.caption(
    "If this page loads without an error, the "
    "Streamlit project can access all required "
    "artifacts correctly."
)
