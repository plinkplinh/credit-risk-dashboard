from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Three-Model Credit Risk PoC",
    page_icon="📊",
    layout="wide",
)


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "artifacts" / "model"
SURVEY_DIR = APP_DIR / "artifacts" / "survey"

MODEL_PATHS = {
    "Logistic regression":
        MODEL_DIR / "logistic_calibrated_baseline.joblib",
    "Explainable Boosting Machine":
        MODEL_DIR / "ebm_calibrated_comparator.joblib",
    "XGBoost":
        MODEL_DIR / "home_credit_benchmark_model.joblib",
}

MODEL_COMPARISON_PATH = MODEL_DIR / "model_comparison_summary.csv"
EXPECTED_COST_PATH = MODEL_DIR / "expected_cost_by_model_and_ratio.csv"
POC_MODEL_SCORES_PATH = MODEL_DIR / "poc_scenario_model_comparison.csv"
POC_MODEL_SUMMARY_PATH = MODEL_DIR / "poc_scenario_model_comparison_summary.csv"
REPRESENTATIVES_PATH = MODEL_DIR / "poc_three_representative_cases.csv"
EBM_LOCAL_PATH = MODEL_DIR / "ebm_local_representative_contributions.csv"
MODEL_GOVERNANCE_PATH = MODEL_DIR / "model_governance_comparison.csv"
PLATT_PARAMETERS_PATH = MODEL_DIR / "platt_calibration_parameters.csv"
SPLIT_DIFFERENCE_PATH = MODEL_DIR / "split_sensitivity_model_difference_summary.csv"
EXT_ABLATION_PATH = MODEL_DIR / "ext_source_ablation.csv"
EXT_SPLIT_SUMMARY_PATH = MODEL_DIR / "split_sensitivity_ext_source_summary.csv"
MANIFEST_PATH = MODEL_DIR / "reproducibility_manifest.json"

SAMPLE_FLOW_PATH = SURVEY_DIR / "sample_flow.csv"
DESCRIPTIVE_RESULTS_PATH = SURVEY_DIR / "descriptive_results.csv"
LIKERT_RESULTS_PATH = SURVEY_DIR / "likert_results.csv"


@st.cache_resource
def load_model_artifacts():
    return {
        model_name: joblib.load(path)
        for model_name, path in MODEL_PATHS.items()
    }


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data
def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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


def flow_value(sample_flow, search_text):
    matches = sample_flow.loc[
        sample_flow["stage"]
        .astype(str)
        .str.contains(search_text, case=False, na=False),
        "n",
    ]
    if matches.empty:
        return None
    return int(matches.iloc[0])


def metric_value(value):
    if value is None:
        return "N/A"
    return f"{value:,}"


def public_manifest(manifest):
    cleaned = json.loads(json.dumps(manifest))
    for record in cleaned.get("input_files", []):
        record.pop("absolute_path", None)
    cleaned.get("analysis_source", {}).pop("absolute_path", None)
    return cleaned


try:
    model_artifacts = load_model_artifacts()
    model_comparison = load_csv(MODEL_COMPARISON_PATH)
    expected_cost = load_csv(EXPECTED_COST_PATH)
    poc_model_scores = load_csv(POC_MODEL_SCORES_PATH)
    poc_model_summary = load_csv(POC_MODEL_SUMMARY_PATH)
    representatives = load_csv(REPRESENTATIVES_PATH)
    ebm_local = load_csv(EBM_LOCAL_PATH)
    model_governance = load_csv(MODEL_GOVERNANCE_PATH)
    platt_parameters = load_csv(PLATT_PARAMETERS_PATH)
    split_differences = load_csv(SPLIT_DIFFERENCE_PATH)
    ext_ablation = load_csv(EXT_ABLATION_PATH)
    ext_split_summary = load_csv(EXT_SPLIT_SUMMARY_PATH)
    manifest = load_json(MANIFEST_PATH)
    sample_flow = load_csv(SAMPLE_FLOW_PATH)
    descriptive_results = load_csv(DESCRIPTIVE_RESULTS_PATH)
    likert_results = load_csv(LIKERT_RESULTS_PATH)
except Exception as error:
    st.error("The application could not load its locked research artifacts.")
    st.exception(error)
    st.stop()


expected_models = {
    "Logistic regression",
    "Explainable Boosting Machine",
    "XGBoost",
}

if set(model_artifacts) != expected_models:
    st.error("The three locked model bundles are incomplete.")
    st.stop()

if not all(isinstance(bundle, dict) for bundle in model_artifacts.values()):
    st.error("A locked model artifact has an unexpected structure.")
    st.stop()

if set(model_comparison["model"]) != expected_models:
    st.error("The three-model comparison table is incomplete.")
    st.stop()

if (
    len(model_comparison) != 3
    or len(poc_model_scores) != 2700
    or len(poc_model_summary) != 9
    or len(representatives) != 3
):
    st.error("The locked output files have unexpected dimensions.")
    st.stop()


xgboost_artifact = model_artifacts["XGBoost"]
xgboost_summary = model_comparison.loc[
    model_comparison["model"].eq("XGBoost")
].iloc[0]

primary_cost_ratio = float(manifest["primary_cost_ratio_fn_to_fp"])
xgboost_threshold = float(xgboost_summary["oof_selected_threshold"])

raw_submissions = flow_value(sample_flow, "Raw Google Forms submissions")
declined_consent = flow_value(sample_flow, "Declined consent")
consenting_sample = flow_value(sample_flow, "Consenting primary sample")
outside_burst = flow_value(sample_flow, "outside flagged")


with st.sidebar:
    st.title("Research PoC")
    st.markdown(
        """
        **Two analytically separate strands**

        1. Three-model Home Credit technical benchmark  
        2. Exploratory consumer survey
        """
    )
    st.warning("Survey records are not used as model inputs.")
    st.caption("University dissertation prototype")


st.title("Alternative-Data Credit Risk Assessment")
st.caption(
    "Locked three-model benchmark outputs and exploratory consumer evidence."
)
st.warning(
    "Research demonstration only. Scores are internally calibrated Home "
    "Credit benchmark outputs, not validated Vietnamese probabilities of "
    "default, automated approval decisions or operational lending advice."
)


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


with overview_tab:
    st.header("Three-model technical benchmark")

    overview_columns = st.columns(4)
    overview_columns[0].metric(
        "Locked test observations",
        f"{int(manifest['test_rows']):,}",
    )
    overview_columns[1].metric("Model comparators", len(model_comparison))
    overview_columns[2].metric(
        "Outer splits",
        int(manifest["split_sensitivity"]["repetitions"]),
    )
    overview_columns[3].metric(
        "Primary FN:FP ratio",
        f"{primary_cost_ratio:.0f}:1",
    )

    comparison_display = model_comparison.rename(
        columns={
            "model": "Model",
            "roc_auc": "ROC-AUC",
            "pr_auc_average_precision": "Average precision",
            "calibrated_brier_score": "Brier score",
            "calibrated_log_loss": "Log loss",
            "oof_selected_threshold": "OOF threshold",
            "test_expected_cost_per_applicant": "Expected cost/applicant",
        }
    )[
        [
            "Model",
            "ROC-AUC",
            "Average precision",
            "Brier score",
            "Log loss",
            "OOF threshold",
            "Expected cost/applicant",
        ]
    ].copy()

    numeric_columns = comparison_display.columns.drop("Model")
    comparison_display[numeric_columns] = comparison_display[
        numeric_columns
    ].round(4)

    st.dataframe(
        comparison_display,
        use_container_width=True,
        hide_index=True,
    )

    primary_cost_results = expected_cost.loc[
        expected_cost["cost_ratio_fn_to_fp"].eq(primary_cost_ratio)
    ].sort_values("test_expected_cost_per_applicant")

    best_cost_row = primary_cost_results.iloc[0]
    next_cost_row = primary_cost_results.iloc[1]
    cost_margin = (
        float(next_cost_row["test_expected_cost_per_applicant"])
        - float(best_cost_row["test_expected_cost_per_applicant"])
    )

    st.info(
        f"At the primary {primary_cost_ratio:.0f}:1 cost ratio, "
        f"{best_cost_row['model']} had the lowest locked-test expected "
        f"cost per applicant. Its margin over the next model was "
        f"{cost_margin:.4f} illustrative cost units."
    )

    st.markdown(
        "Discrimination, calibration, expected cost and governance must be "
        "considered together; the highest AUC does not by itself select an "
        "operational model."
    )

    diagnostic_figures = {
        "ROC and precision-recall curves":
            MODEL_DIR / "Figure_M1_ROC_PR_Curves.png",
        "Calibration comparison":
            MODEL_DIR / "Figure_M2_Calibration.png",
        "Expected-cost curves":
            MODEL_DIR / "Figure_M7_Expected_Cost_Curves.png",
        "Split sensitivity — not temporal validation":
            MODEL_DIR / "Figure_M8_Split_Sensitivity.png",
        "OOF threshold stability":
            MODEL_DIR / "Figure_M9_Threshold_Stability.png",
        "EBM intrinsic global explanation":
            MODEL_DIR / "Figure_M10_EBM_Global_Explanation.png",
        "XGBoost post-hoc TreeSHAP":
            MODEL_DIR / "Figure_M3_SHAP_Global.png",
    }

    available_diagnostics = {
        label: path
        for label, path in diagnostic_figures.items()
        if path.exists()
    }

    selected_diagnostic = st.selectbox(
        "Select a diagnostic",
        list(available_diagnostics),
        key="diagnostic_selector",
    )
    st.image(
        str(available_diagnostics[selected_diagnostic]),
        caption=selected_diagnostic,
        use_container_width=True,
    )

    st.caption(
        "Figure M0 is intentionally not displayed until its locked-test "
        "evaluation arrows have been corrected."
    )


with scenarios_tab:
    st.header("Three-model proof-of-concept stress scenarios")
    st.write(
        "Each scenario contains 300 controlled perturbations of sampled "
        "benchmark cases. Quantile values were learned from the training "
        "partition. The scenarios are not external validation."
    )

    median_chart = (
        poc_model_summary
        .pivot(index="scenario", columns="model", values="median")
        * 100
    )
    st.subheader("Median calibrated score by model")
    st.bar_chart(median_chart, use_container_width=True)

    selected_scenario_model = st.selectbox(
        "Select a model",
        sorted(poc_model_summary["model"].unique()),
        key="scenario_model_selector",
    )

    selected_model_summary = poc_model_summary.loc[
        poc_model_summary["model"].eq(selected_scenario_model)
    ].copy()

    display_summary = pd.DataFrame(
        {
            "Scenario": selected_model_summary["scenario"],
            "N": selected_model_summary["n"].astype(int),
            "Mean": selected_model_summary["mean"].map(percentage),
            "Median": selected_model_summary["median"].map(percentage),
            "IQR": [
                f"{percentage(q25)}–{percentage(q75)}"
                for q25, q75 in zip(
                    selected_model_summary["q25"],
                    selected_model_summary["q75"],
                )
            ],
            "Above own OOF threshold": selected_model_summary[
                "proportion_above_illustrative_threshold"
            ].map(percentage),
        }
    )

    st.dataframe(
        display_summary,
        use_container_width=True,
        hide_index=True,
    )

    selected_scenario = st.selectbox(
        "Inspect one scenario",
        selected_model_summary["scenario"].tolist(),
        key="scenario_distribution_selector",
    )
    selected_summary = selected_model_summary.loc[
        selected_model_summary["scenario"].eq(selected_scenario)
    ].iloc[0]

    scenario_columns = st.columns(5)
    scenario_columns[0].metric("Observations", int(selected_summary["n"]))
    scenario_columns[1].metric("Mean", percentage(selected_summary["mean"]))
    scenario_columns[2].metric("Median", percentage(selected_summary["median"]))
    scenario_columns[3].metric(
        "IQR",
        f"{percentage(selected_summary['q25'])} – "
        f"{percentage(selected_summary['q75'])}",
    )
    scenario_columns[4].metric(
        "Above threshold",
        percentage(
            selected_summary["proportion_above_illustrative_threshold"]
        ),
    )

    st.info(
        "Above-threshold proportions use each model's own OOF-selected "
        "threshold at FN:FP = 5:1. They are not approval rates."
    )


with case_tab:
    st.header("Representative median-score cases")

    selected_case_scenario = st.selectbox(
        "Select a representative scenario",
        representatives["scenario"].astype(str).tolist(),
        key="representative_case_selector",
    )
    selected_case = representatives.loc[
        representatives["scenario"].eq(selected_case_scenario)
    ].iloc[0]
    case_position = representatives.index[
        representatives["scenario"].eq(selected_case_scenario)
    ].tolist()[0]

    case_model_scores = poc_model_scores.loc[
        poc_model_scores["scenario_case_id"].eq(
            selected_case["scenario_case_id"]
        ),
        [
            "model",
            "raw_score",
            "internally_calibrated_probability",
            "above_illustrative_cost_threshold",
        ],
    ].copy()
    case_model_scores["raw_score"] = case_model_scores["raw_score"].map(
        percentage
    )
    case_model_scores["internally_calibrated_probability"] = (
        case_model_scores["internally_calibrated_probability"].map(percentage)
    )
    case_model_scores = case_model_scores.rename(
        columns={
            "model": "Model",
            "raw_score": "Raw model score",
            "internally_calibrated_probability": "Calibrated score",
            "above_illustrative_cost_threshold": "Above own threshold",
        }
    )

    case_columns = st.columns(2)
    case_columns[0].metric(
        "Case ID",
        str(selected_case["scenario_case_id"]),
    )
    case_columns[1].metric(
        "XGBoost OOF threshold",
        f"{xgboost_threshold:.4f}",
    )

    st.subheader("Locked three-model scores")
    st.dataframe(
        case_model_scores,
        use_container_width=True,
        hide_index=True,
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
                dataset_number(selected_case["AMT_INCOME_TOTAL"]),
                dataset_number(selected_case["AMT_CREDIT"]),
                (
                    "Missing"
                    if pd.isna(selected_case["CREDIT_INCOME_RATIO"])
                    else f"{selected_case['CREDIT_INCOME_RATIO']:.3f}"
                ),
                external_score(selected_case["EXT_SOURCE_1"]),
                external_score(selected_case["EXT_SOURCE_2"]),
                external_score(selected_case["EXT_SOURCE_3"]),
            ],
        }
    )

    st.subheader("Displayed benchmark variables")
    st.dataframe(profile_table, use_container_width=True, hide_index=True)
    st.caption(
        "Amounts are shown in dataset-native units. They are not converted "
        "to VND or interpreted as a real Vietnamese applicant."
    )

    explanation_type = st.radio(
        "Explanation method",
        [
            "XGBoost TreeSHAP",
            "EBM intrinsic local contributions",
        ],
        horizontal=True,
    )

    if explanation_type == "XGBoost TreeSHAP":
        shap_path = MODEL_DIR / (
            f"Figure_M{4 + case_position}_SHAP_Representative_"
            f"{case_position + 1}.png"
        )
        st.image(
            str(shap_path),
            caption=f"Post-hoc TreeSHAP — {selected_case_scenario}",
            use_container_width=True,
        )
        st.info(
            "TreeSHAP describes the fitted XGBoost raw margin. It does not "
            "explain the later Platt mapping, establish causality or provide "
            "a legal lending reason code."
        )
    else:
        representative_number = case_position + 1
        ebm_case = ebm_local.loc[
            ebm_local["representative_case"].eq(representative_number)
        ].copy()
        ebm_case["absolute_contribution"] = ebm_case[
            "raw_log_odds_contribution"
        ].abs()
        top_ebm_terms = (
            ebm_case
            .nlargest(12, "absolute_contribution")
            .sort_values("raw_log_odds_contribution")
        )
        st.bar_chart(
            top_ebm_terms.set_index("term")[["raw_log_odds_contribution"]],
            use_container_width=True,
        )
        st.info(
            "EBM contributions are intrinsic additive effects on the raw "
            "default log-odds scale. Platt calibration remains a separate "
            "probability mapping."
        )
        with st.expander("View combined EBM local-explanation figure"):
            st.image(
                str(MODEL_DIR / "Figure_M11_EBM_Local_Explanations.png"),
                use_container_width=True,
            )


with survey_tab:
    st.header("Exploratory consumer survey")

    survey_columns = st.columns(4)
    survey_columns[0].metric("Raw submissions", metric_value(raw_submissions))
    survey_columns[1].metric("Declined consent", metric_value(declined_consent))
    survey_columns[2].metric("Consenting sample", metric_value(consenting_sample))
    survey_columns[3].metric(
        "Outside flagged interval",
        metric_value(outside_burst),
    )

    st.dataframe(sample_flow, use_container_width=True, hide_index=True)

    survey_figures = {
        "Data-sharing motivations":
            SURVEY_DIR / "Figure_4_6_Sharing_Motivations.png",
        "Required interest discount":
            SURVEY_DIR / "Figure_4_7_Interest_Discount.png",
        "Data-sharing red lines":
            SURVEY_DIR / "Figure_4_8_Data_Sharing_Red_Lines.png",
        "Institutional trust":
            SURVEY_DIR / "Figure_4_9_Institutional_Trust.png",
        "Likert distributions":
            SURVEY_DIR / "Figure_4_10_Likert_Distributions.png",
        "Recruitment-wave sensitivity":
            SURVEY_DIR / "Figure_S1_Burst_Sensitivity.png",
    }
    selected_survey_figure = st.selectbox(
        "Select an aggregate survey figure",
        list(survey_figures),
        key="survey_figure_selector",
    )
    st.image(
        str(survey_figures[selected_survey_figure]),
        caption=selected_survey_figure,
        use_container_width=True,
    )

    with st.expander("View aggregate survey result tables"):
        st.markdown("#### Descriptive results")
        st.dataframe(
            descriptive_results,
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("#### Likert results")
        st.dataframe(
            likert_results,
            use_container_width=True,
            hide_index=True,
        )

    st.warning(
        "The survey used convenience and snowball recruitment. Responses "
        "are stated preferences, not nationally representative prevalence "
        "estimates or observed disclosure behaviour."
    )


with limitations_tab:
    st.header("Methods, governance and limitations")

    st.subheader("Locked analytical configuration")
    configuration_columns = st.columns(4)
    configuration_columns[0].metric("Loaded model bundles", len(model_artifacts))
    configuration_columns[1].metric("OOF calibration folds", manifest["cv_folds"])
    configuration_columns[2].metric(
        "Split repetitions",
        manifest["split_sensitivity"]["repetitions"],
    )
    configuration_columns[3].metric(
        "EBM interactions",
        manifest["ebm_configuration"]["interactions"],
    )

    st.info(
        xgboost_artifact.get(
            "claim_boundary",
            "Technical benchmark only; not a validated Vietnamese "
            "underwriting system.",
        )
    )

    st.markdown(
        f"""
        - Training observations: `{manifest['train_rows']:,}`
        - Locked test observations: `{manifest['test_rows']:,}`
        - Primary outer-split seed: `{manifest['primary_outer_split_seed']}`
        - Cost ratios tested: `{manifest['cost_ratios_tested']}`
        - All three comparators used class balancing.
        - Each comparator received a separate Platt mapping fitted to five-fold OOF training predictions.
        - Thresholds were selected from calibrated OOF training predictions.
        """
    )

    st.subheader("Model-governance comparison")
    st.dataframe(
        model_governance,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("OOF Platt calibration")
    st.dataframe(
        platt_parameters,
        use_container_width=True,
        hide_index=True,
    )

    selected_cost_ratio = st.select_slider(
        "Select illustrative FN:FP cost ratio",
        options=sorted(expected_cost["cost_ratio_fn_to_fp"].unique()),
        value=primary_cost_ratio,
    )
    selected_cost_results = expected_cost.loc[
        expected_cost["cost_ratio_fn_to_fp"].eq(selected_cost_ratio),
        [
            "model",
            "threshold",
            "test_expected_cost_per_applicant",
            "recall_sensitivity",
            "specificity",
            "test_cost_rank_within_ratio",
        ],
    ].sort_values("test_cost_rank_within_ratio")
    st.dataframe(
        selected_cost_results.round(4),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "The ratios are illustrative relative weights, not observed "
        "Vietnamese monetary losses or commercially optimal policies."
    )

    with st.expander("View 30-split model-difference summary"):
        st.dataframe(
            split_differences.round(4),
            use_container_width=True,
            hide_index=True,
        )
        st.warning(
            "These repeated stratified outer splits assess random split "
            "sensitivity only. They are not temporal validation because "
            "usable application dates were unavailable."
        )

    with st.expander("View EXT_SOURCE ablation evidence"):
        st.markdown("#### Primary locked split")
        st.dataframe(ext_ablation, use_container_width=True, hide_index=True)
        st.markdown("#### Thirty-split gap summary")
        st.dataframe(
            ext_split_summary.round(4),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Interpretation safeguards")
    st.markdown(
        """
        - Home Credit is a public technical benchmark and is not assumed to contain Vietnamese borrowers.
        - Logistic regression and the main-effects-only EBM are intrinsically interpretable; XGBoost uses post-hoc TreeSHAP.
        - Intrinsic or post-hoc explanations do not establish causality, fairness, legitimacy or legal compliance.
        - `EXT_SOURCE_1/2/3` are opaque external scores and are not relabelled as CIC, e-wallet, e-commerce or social-media variables.
        - Scenario cases are controlled perturbations, not real applicants or external validation.
        - Model outputs do not constitute automated approval or rejection decisions.
        - Survey and model records are never joined; integration occurs only during interpretation.
        - The prototype is not connected to CIC, a bank, a P2P platform or live consumer data.
        """
    )

    with st.expander("View public reproducibility manifest"):
        st.json(public_manifest(manifest))


st.divider()
st.caption(
    "Dissertation research prototype — locked benchmark outputs and "
    "aggregate exploratory survey evidence."
)
