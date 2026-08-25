
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Alternative-Data Credit Risk PoC",
    page_icon="📊",
    layout="wide",
)


# =====================================================
# PATHS
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

REPRESENTATIVES_PATH = (
    MODEL_DIR /
    "poc_three_representative_cases.csv"
)

MANIFEST_PATH = (
    MODEL_DIR /
    "reproducibility_manifest.json"
)

SAMPLE_FLOW_PATH = (
    SURVEY_DIR /
    "sample_flow.csv"
)

DESCRIPTIVE_RESULTS_PATH = (
    SURVEY_DIR /
    "descriptive_results.csv"
)

LIKERT_RESULTS_PATH = (
    SURVEY_DIR /
    "likert_results.csv"
)


# =====================================================
# LOADERS
# =====================================================

@st.cache_resource
def load_model_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data
def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def percentage(value, digits=2):
    if pd.isna(value):
        return "Missing"
    return f"{float(value):.{digits}%}"


def dataset_number(value):
    if pd.isna(value):
        return "Missing"
    return f"{float(value):,.0f}"


def external_score(value):
    if pd.isna(value):
        return "Missing"
    return f"{float(value):.3f}"


def flow_value(search_text):
    matches = sample_flow.loc[
        sample_flow["stage"]
        .astype(str)
        .str.contains(
            search_text,
            case=False,
            na=False,
        ),
        "n",
    ]

    if matches.empty:
        return None

    return int(matches.iloc[0])


def metric_value(value):
    if value is None:
        return "N/A"
    return f"{value:,}"


# =====================================================
# LOAD ARTIFACTS
# =====================================================

try:
    model_artifact = load_model_artifact()

    model_metrics = load_csv(METRICS_PATH)
    poc_scores = load_csv(POC_SCORES_PATH)
    poc_summary = load_csv(POC_SUMMARY_PATH)

    representatives = load_csv(
        REPRESENTATIVES_PATH
    )

    manifest = load_json(MANIFEST_PATH)

    sample_flow = load_csv(SAMPLE_FLOW_PATH)

    descriptive_results = load_csv(
        DESCRIPTIVE_RESULTS_PATH
    )

    likert_results = load_csv(
        LIKERT_RESULTS_PATH
    )

except Exception as error:
    st.error(
        "The application could not load its "
        "audited research artifacts."
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

missing_keys = (
    required_model_keys
    .difference(model_artifact.keys())
)

if missing_keys:
    st.error(
        "The model bundle is missing: "
        + ", ".join(sorted(missing_keys))
    )
    st.stop()


if (
    len(poc_scores) != 900
    or len(poc_summary) != 3
    or len(representatives) != 3
):
    st.error(
        "The PoC files do not match the final "
        "900-observation protocol."
    )
    st.stop()


# =====================================================
# DERIVED VALUES
# =====================================================

preferred_metric = model_metrics.loc[
    model_metrics["model"].eq(
        "XGBoost calibrated / cost threshold"
    )
]

if preferred_metric.empty:
    selected_metric = model_metrics.iloc[-1]
else:
    selected_metric = preferred_metric.iloc[0]


decision_threshold = float(
    model_artifact["decision_threshold"]
)

scenario_names = (
    poc_summary["scenario"]
    .astype(str)
    .tolist()
)

raw_submissions = flow_value(
    "Raw Google Forms submissions"
)

declined_consent = flow_value(
    "Declined consent"
)

consenting_sample = flow_value(
    "Consenting primary sample"
)

outside_burst = flow_value(
    "outside flagged"
)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.title("Research PoC")

    st.markdown(
        """
        **Two analytically separate strands**

        1. Home Credit technical benchmark  
        2. Exploratory consumer survey
        """
    )

    st.warning(
        "The survey is not used as model input."
    )

    st.caption(
        "University dissertation prototype"
    )


# =====================================================
# HEADER
# =====================================================

st.title(
    "Alternative-Data Credit Risk Assessment"
)

st.caption(
    "Interactive proof of concept based on an "
    "audited technical benchmark and an "
    "exploratory consumer survey."
)

st.warning(
    "Research demonstration only. Scores are "
    "internally calibrated Home Credit benchmark "
    "outputs, not validated Vietnamese probabilities "
    "of default or operational lending decisions."
)


# =====================================================
# NAVIGATION
# =====================================================

(
    overview_tab,
    scenarios_tab,
    case_tab,
    survey_tab,
    limitations_tab,
) = st.tabs(
    [
        "Overview",
        "Scenario comparison",
        "Representative case",
        "Survey evidence",
        "Methods and limitations",
    ]
)


# =====================================================
# TAB 1 — OVERVIEW
# =====================================================

with overview_tab:
    st.header("Technical benchmark overview")

    metric_columns = st.columns(5)

    metric_columns[0].metric(
        "Test observations",
        f"{int(selected_metric['n']):,}",
    )

    metric_columns[1].metric(
        "ROC-AUC",
        f"{selected_metric['roc_auc']:.4f}",
    )

    metric_columns[2].metric(
        "PR-AUC",
        f"{selected_metric['pr_auc_average_precision']:.4f}",
    )

    metric_columns[3].metric(
        "Brier score",
        f"{selected_metric['brier_score']:.4f}",
    )

    metric_columns[4].metric(
        "Illustrative threshold",
        f"{decision_threshold:.4f}",
    )

    st.caption(
        "The threshold was selected using illustrative "
        "false-negative and false-positive costs. It is "
        "not a commercial lending threshold."
    )

    st.subheader("Research components")

    component_columns = st.columns(4)

    component_columns[0].metric(
        "Raw predictors",
        len(
            model_artifact[
                "raw_predictor_columns"
            ]
        ),
    )

    component_columns[1].metric(
        "PoC observations",
        len(poc_scores),
    )

    component_columns[2].metric(
        "Scenario cohorts",
        len(poc_summary),
    )

    component_columns[3].metric(
        "Survey consenters",
        metric_value(consenting_sample),
    )

    st.markdown(
        """
        The model strand evaluates technical discrimination,
        calibration, threshold sensitivity and local
        explanations on the public Home Credit benchmark.

        The survey strand describes stated data-sharing
        incentives, privacy boundaries and institutional
        trust among consenting respondents. The two datasets
        are integrated only at the interpretation stage.
        """
    )

    diagnostic_figures = {
        "ROC and precision-recall curves":
            MODEL_DIR / "Figure_M1_ROC_PR_Curves.png",

        "Internal calibration":
            MODEL_DIR / "Figure_M2_Calibration.png",

        "Global SHAP summary":
            MODEL_DIR / "Figure_M3_SHAP_Global.png",
    }

    available_diagnostics = {
        label: path
        for label, path
        in diagnostic_figures.items()
        if path.exists()
    }

    if available_diagnostics:
        st.subheader("Model diagnostic figure")

        selected_diagnostic = st.selectbox(
            "Select a diagnostic",
            list(available_diagnostics.keys()),
            key="diagnostic_selector",
        )

        st.image(
            str(
                available_diagnostics[
                    selected_diagnostic
                ]
            ),
            caption=selected_diagnostic,
            use_container_width=True,
        )


# =====================================================
# TAB 2 — SCENARIO COMPARISON
# =====================================================

with scenarios_tab:
    st.header(
        "Proof-of-concept stress scenarios"
    )

    st.write(
        "Each scenario contains 300 controlled "
        "perturbations of sampled benchmark cases. "
        "Quantile values were learned from the "
        "training partition."
    )

    display_summary = pd.DataFrame(
        {
            "Scenario": poc_summary["scenario"],
            "N": poc_summary["n"].astype(int),
            "Median score": (
                poc_summary["median"]
                .map(percentage)
            ),
            "IQR": [
                (
                    f"{percentage(q25)}–"
                    f"{percentage(q75)}"
                )
                for q25, q75
                in zip(
                    poc_summary["q25"],
                    poc_summary["q75"],
                )
            ],
            "5th–95th percentile": [
                (
                    f"{percentage(q05)}–"
                    f"{percentage(q95)}"
                )
                for q05, q95
                in zip(
                    poc_summary["q05"],
                    poc_summary["q95"],
                )
            ],
            "Above illustrative threshold": (
                poc_summary[
                    "proportion_above_illustrative_threshold"
                ]
                .map(percentage)
            ),
        }
    )

    st.dataframe(
        display_summary,
        use_container_width=True,
        hide_index=True,
    )

    median_chart = (
        poc_summary[
            ["scenario", "median"]
        ]
        .assign(
            median=lambda frame:
                frame["median"] * 100
        )
        .rename(
            columns={
                "scenario": "Scenario",
                "median": "Median score (%)",
            }
        )
        .set_index("Scenario")
    )

    st.subheader(
        "Median internally calibrated score"
    )

    st.bar_chart(
        median_chart,
        use_container_width=True,
    )

    selected_scenario = st.selectbox(
        "Inspect one scenario distribution",
        scenario_names,
        key="scenario_distribution_selector",
    )

    selected_summary = poc_summary.loc[
        poc_summary["scenario"].eq(
            selected_scenario
        )
    ].iloc[0]

    distribution_columns = st.columns(5)

    distribution_columns[0].metric(
        "Observations",
        f"{int(selected_summary['n']):,}",
    )

    distribution_columns[1].metric(
        "Mean",
        percentage(
            selected_summary["mean"]
        ),
    )

    distribution_columns[2].metric(
        "Median",
        percentage(
            selected_summary["median"]
        ),
    )

    distribution_columns[3].metric(
        "5th–95th range",
        (
            f"{percentage(selected_summary['q05'])}"
            " – "
            f"{percentage(selected_summary['q95'])}"
        ),
    )

    distribution_columns[4].metric(
        "Above threshold",
        percentage(
            selected_summary[
                "proportion_above_illustrative_threshold"
            ]
        ),
    )

    st.info(
        "Differences demonstrate scenario responsiveness "
        "within the fitted benchmark model. They do not "
        "establish external predictive validity."
    )


# =====================================================
# TAB 3 — REPRESENTATIVE CASE
# =====================================================

with case_tab:
    st.header(
        "Representative median-score case"
    )

    selected_case_scenario = st.selectbox(
        "Select a representative scenario",
        representatives[
            "scenario"
        ].astype(str).tolist(),
        key="representative_case_selector",
    )

    selected_case = representatives.loc[
        representatives["scenario"].eq(
            selected_case_scenario
        )
    ].iloc[0]

    case_position = (
        representatives.index[
            representatives["scenario"].eq(
                selected_case_scenario
            )
        ]
        .tolist()[0]
    )

    above_threshold_value = selected_case[
        "above_illustrative_cost_threshold"
    ]

    if isinstance(
        above_threshold_value,
        str,
    ):
        is_above_threshold = (
            above_threshold_value
            .strip()
            .lower()
            == "true"
        )
    else:
        is_above_threshold = bool(
            above_threshold_value
        )

    threshold_comparison = (
        "Above"
        if is_above_threshold
        else "Below"
    )

    case_metrics = st.columns(4)

    case_metrics[0].metric(
        "Case ID",
        str(
            selected_case[
                "scenario_case_id"
            ]
        ),
    )

    case_metrics[1].metric(
        "Calibrated benchmark score",
        percentage(
            selected_case[
                "internally_calibrated_probability"
            ]
        ),
    )

    case_metrics[2].metric(
        "Raw XGBoost score",
        percentage(
            selected_case[
                "raw_xgboost_score"
            ]
        ),
    )

    case_metrics[3].metric(
        "Illustrative threshold",
        threshold_comparison,
    )

    profile_table = pd.DataFrame(
        {
            "Variable": [
                "Income total",
                "Credit amount",
                "Credit-to-income ratio",
                "EXT_SOURCE_1",
                "EXT_SOURCE_2",
                "EXT_SOURCE_3",
            ],
            "Value": [
                dataset_number(
                    selected_case[
                        "AMT_INCOME_TOTAL"
                    ]
                ),
                dataset_number(
                    selected_case[
                        "AMT_CREDIT"
                    ]
                ),
                (
                    "Missing"
                    if pd.isna(
                        selected_case[
                            "CREDIT_INCOME_RATIO"
                        ]
                    )
                    else (
                        f"{selected_case['CREDIT_INCOME_RATIO']:.3f}"
                    )
                ),
                external_score(
                    selected_case[
                        "EXT_SOURCE_1"
                    ]
                ),
                external_score(
                    selected_case[
                        "EXT_SOURCE_2"
                    ]
                ),
                external_score(
                    selected_case[
                        "EXT_SOURCE_3"
                    ]
                ),
            ],
        }
    )

    st.subheader(
        "Displayed benchmark variables"
    )

    st.dataframe(
        profile_table,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Amounts are shown in dataset-native units. "
        "They are not converted to VND or interpreted "
        "as a real Vietnamese applicant."
    )

    shap_path = (
        MODEL_DIR /
        (
            f"Figure_M{4 + case_position}_"
            f"SHAP_Representative_"
            f"{case_position + 1}.png"
        )
    )

    st.subheader(
        "Local model explanation"
    )

    if shap_path.exists():
        st.image(
            str(shap_path),
            caption=(
                "TreeSHAP explanation for "
                f"{selected_case_scenario}"
            ),
            use_container_width=True,
        )
    else:
        st.warning(
            f"SHAP figure not found: "
            f"{shap_path.name}"
        )

    st.info(
        "The waterfall explains the raw XGBoost "
        "model margin. It does not explain the "
        "subsequent Platt calibration mapping, "
        "establish causality or constitute a legal "
        "lending reason code."
    )


# =====================================================
# TAB 4 — SURVEY EVIDENCE
# =====================================================

with survey_tab:
    st.header(
        "Exploratory consumer survey"
    )

    survey_columns = st.columns(4)

    survey_columns[0].metric(
        "Raw submissions",
        metric_value(raw_submissions),
    )

    survey_columns[1].metric(
        "Declined consent",
        metric_value(declined_consent),
    )

    survey_columns[2].metric(
        "Primary sample",
        metric_value(consenting_sample),
    )

    survey_columns[3].metric(
        "Outside flagged interval",
        metric_value(outside_burst),
    )

    st.dataframe(
        sample_flow,
        use_container_width=True,
        hide_index=True,
    )

    survey_figures = {
        "Data-sharing motivations":
            SURVEY_DIR /
            "Figure_4_6_Sharing_Motivations.png",

        "Required interest discount":
            SURVEY_DIR /
            "Figure_4_7_Interest_Discount.png",

        "Data-sharing red lines":
            SURVEY_DIR /
            "Figure_4_8_Data_Sharing_Red_Lines.png",

        "Institutional trust":
            SURVEY_DIR /
            "Figure_4_9_Institutional_Trust.png",

        "Likert distributions":
            SURVEY_DIR /
            "Figure_4_10_Likert_Distributions.png",

        "Recruitment-wave sensitivity":
            SURVEY_DIR /
            "Figure_S1_Burst_Sensitivity.png",
    }

    available_survey_figures = {
        label: path
        for label, path
        in survey_figures.items()
        if path.exists()
    }

    selected_survey_figure = st.selectbox(
        "Select an aggregate survey figure",
        list(
            available_survey_figures.keys()
        ),
        key="survey_figure_selector",
    )

    st.image(
        str(
            available_survey_figures[
                selected_survey_figure
            ]
        ),
        caption=selected_survey_figure,
        use_container_width=True,
    )

    with st.expander(
        "View aggregate survey result tables"
    ):
        st.markdown(
            "#### Descriptive results"
        )

        st.dataframe(
            descriptive_results,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            "#### Likert results"
        )

        st.dataframe(
            likert_results,
            use_container_width=True,
            hide_index=True,
        )

    st.warning(
        "The survey used convenience and snowball "
        "recruitment. Responses are stated preferences, "
        "not nationally representative prevalence "
        "estimates or observed disclosure behaviour."
    )


# =====================================================
# TAB 5 — METHODS AND LIMITATIONS
# =====================================================

with limitations_tab:
    st.header(
        "Methods, safeguards and limitations"
    )

    st.subheader(
        "Model governance boundary"
    )

    st.info(
        model_artifact["claim_boundary"]
    )

    cost_assumptions = model_artifact.get(
        "threshold_cost_assumptions",
        {},
    )

    st.markdown(
        f"""
        **Final analytical configuration**

        - Training observations: `{manifest.get('train_rows', 'N/A')}`
        - Locked test observations: `{manifest.get('test_rows', 'N/A')}`
        - Cross-validation folds: `{manifest.get('cv_folds', 'N/A')}`
        - Random seed: `{manifest.get('random_state', 'N/A')}`
        - PoC base cases per scenario: `{manifest.get('poc_base_cases_per_scenario', 'N/A')}`
        - False-negative analytical weight: `{cost_assumptions.get('false_negative_cost', 'N/A')}`
        - False-positive analytical weight: `{cost_assumptions.get('false_positive_cost', 'N/A')}`
        """
    )

    st.subheader(
        "Interpretation safeguards"
    )

    st.markdown(
        """
        - The Home Credit dataset is treated as a public technical benchmark.
        - The benchmark is not assumed to contain Vietnamese borrowers.
        - EXT_SOURCE variables are opaque external scores and are not labelled as VietQR, e-wallet or e-commerce data.
        - Scenario cases are controlled perturbations, not financial statements or real applicants.
        - The threshold is illustrative and is not an approve/reject rule.
        - SHAP explains fitted model behaviour, not causality, fairness or legal compliance.
        - Survey records are never joined to model records.
        - Survey evidence is exploratory and affected by non-probability recruitment and a concentrated submission wave.
        - The prototype is not connected to CIC, a bank, a P2P platform or live consumer data.
        """
    )

    with st.expander(
        "View reproducibility manifest"
    ):
        st.json(manifest)


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Dissertation research prototype — audited "
    "benchmark outputs and aggregate survey evidence."
)
