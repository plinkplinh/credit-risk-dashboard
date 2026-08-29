# Alternative-Data Credit Risk Assessment

Streamlit proof of concept for an MSc dissertation. The application
presents two analytically separate evidence strands:

1. A locked Home Credit technical benchmark comparing class-balanced
   logistic regression, a main-effects-only Explainable Boosting Machine
   and XGBoost.
2. An exploratory consumer survey about data-sharing incentives, privacy
   boundaries and institutional trust.

Survey records are never used as model predictors or joined to model data.

## Run locally

Use Python 3.13 for the closest compatibility with the saved artifacts.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app_smoke_test.py
python -m streamlit run app.py
```

## Analytical design

- Three class-balanced model comparators.
- Separate five-fold OOF Platt calibration for every model.
- Threshold selection from calibrated OOF training predictions.
- Expected cost per applicant at FN:FP ratios 1:1, 2:1, 5:1, 10:1 and 20:1.
- Thirty repeated stratified outer splits reported as split sensitivity,
  not temporal validation.
- EBM intrinsic global/local explanations and XGBoost post-hoc TreeSHAP.

## Evidence boundary

This is a presentation-only research demonstration. It is not a validated
Vietnamese probability-of-default system, live underwriting service,
automated approval tool or operational lending recommendation.
