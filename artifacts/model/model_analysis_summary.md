# Credit-model benchmark audit summary

## Data and split

- Application rows: **307,511**; predictors after optional aggregation/engineering: **124**.
- Independent stratified test set: **61,503 (20%)**; never used for preprocessing fit, tuning, calibration fit, or threshold selection.
- Training default/non-default ratio produced scale_pos_weight **11.3871**.
- Grid search: **16 candidates x 5 stratified folds**; refit criterion = average precision (PR-AUC).
- Auxiliary tables used: **none; application table only**.

## Selected hold-out result

- ROC-AUC: **0.7687**; PR-AUC/AP: **0.2588**.
- Precision: **0.2662**; recall: **0.3925**; F1: **0.3173**.
- Specificity: **0.9050**; balanced accuracy: **0.6488**; MCC: **0.2503**.
- Brier score: **0.0671**; threshold: **0.1699**.
- Confusion counts (TN/FP/FN/TP): **51166/5372/3016/1949**.

## Threshold assumption

- Threshold selected from training out-of-fold calibrated scores using illustrative costs FN=5.00 and FP=1.00.
- These are scenario weights, not empirically estimated Vietnamese loan-loss amounts. Chapter 4 must include threshold sensitivity rather than calling this threshold commercially optimal.


## Fair raw and calibrated comparison

- Weighted logistic regression: Brier **0.2025 raw** and **0.0684 after Platt calibration**.
- Weighted XGBoost: Brier **0.1797 raw** and **0.0671 after Platt calibration**.
- The calibrated Brier difference between logistic regression and XGBoost was **0.0013**.
- The previously large raw-logistic versus calibrated-XGBoost difference must not be attributed solely to model class because it combined model and calibration effects.

## Bootstrap uncertainty

- roc_auc: **0.7687** (95% bootstrap CI **0.7623-0.7752**).
- pr_auc_average_precision: **0.2588** (95% bootstrap CI **0.2484-0.2698**).
- precision: **0.2662** (95% bootstrap CI **0.2575-0.2749**).
- recall_sensitivity: **0.3925** (95% bootstrap CI **0.3795-0.4070**).
- specificity: **0.9050** (95% bootstrap CI **0.9027-0.9075**).
- f1: **0.3173** (95% bootstrap CI **0.3073-0.3276**).
- balanced_accuracy: **0.6488** (95% bootstrap CI **0.6421-0.6559**).
- mcc: **0.2503** (95% bootstrap CI **0.2391-0.2620**).
- brier_score: **0.0671** (95% bootstrap CI **0.0666-0.0675**).

## EXT_SOURCE ablation

- Full XGBoost: ROC-AUC **0.7687**; PR-AUC/AP **0.2588**.
- XGBoost without EXT_SOURCE_1/2/3: ROC-AUC **0.7196**; PR-AUC/AP **0.1947**.
- XGBoost using EXT_SOURCE columns only: ROC-AUC **0.7208**; PR-AUC/AP **0.2069**.

## PoC stress scenarios

- A: higher external scores / lower credit-income ratio: N=300, median internally calibrated score **0.013**, IQR **0.009-0.018**.
- B: EXT_SOURCE_1 missing / moderate other scores: N=300, median internally calibrated score **0.034**, IQR **0.022-0.052**.
- C: lower external scores / higher credit-income ratio: N=300, median internally calibrated score **0.126**, IQR **0.086-0.181**.

## Interpretation limits required in the dissertation

- Home Credit is a public technical benchmark/proxy; record country and platform provenance are undisclosed and external validity for Vietnam is not established.
- EXT_SOURCE_1/2/3 are undocumented external scores. The analysis cannot identify them as CIC, e-commerce, e-wallet, socio-demographic, or direct digital-footprint measures.
- Internal calibration improves agreement on this benchmark but does not make scores operational Vietnamese probabilities of default.
- SHAP explains how this fitted model used benchmark variables; it does not establish causality, legitimacy, fairness, or regulatory compliance.
- Subgroup metrics are an error audit, not proof of fairness. Small cells and unknown group semantics must be disclosed.
- The PoC is a controlled sensitivity exercise. It is not a real-time Vietnamese bank-statement pipeline, external validation, or evidence for automated approval/rejection.
- Any Streamlit interface should be described as a separate demonstration only if its source and deployment evidence are actually submitted.
