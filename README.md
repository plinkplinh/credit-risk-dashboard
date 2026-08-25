# Alternative-Data Credit Risk Assessment

Interactive dissertation proof of concept developed with
Streamlit.

## Research structure

The application presents two analytically separate strands:

1. A technical credit-risk benchmark based on the public
   Home Credit dataset.
2. An exploratory consumer survey concerning data-sharing
   incentives, privacy boundaries and institutional trust.

Survey responses are not used as predictors and survey records
are never joined to model records.

## Run locally

For the closest compatibility with the saved model artifact,
use Python 3.13.5.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Application sections

- Overview
- Scenario comparison
- Representative case and local SHAP explanation
- Aggregate survey evidence
- Methods, safeguards and limitations

## Evidence boundary

This application is a research demonstration only. The scores
are internally calibrated Home Credit benchmark outputs. They
are not validated Vietnamese probabilities of default, live
credit decisions or operational lending recommendations.

The scenario cases are controlled perturbations rather than
real applicants. SHAP values explain fitted model behaviour;
they do not establish causality, fairness, legal compliance or
formal adverse-action reason codes.

## Reproducibility

Model configuration is recorded in:

```text
artifacts/model/reproducibility_manifest.json
```

Final package validation is recorded in:

```text
qa_report.json
```
