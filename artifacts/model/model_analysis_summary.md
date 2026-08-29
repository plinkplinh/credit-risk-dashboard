# Three-model credit benchmark audit summary

## Data, split and model lock

- Application rows: **307,511**; predictors: **124**.
- Primary independent stratified test set: **61,503 (20%)**; never used for preprocessing fit, tuning, calibration fit or threshold selection.
- XGBoost grid search: **16 candidates x 5 folds**; refit criterion = average precision.
- The EBM was pre-specified as a main-effects-only glass-box comparator (interactions=0); it was not interpreted through SHAP.
- All three comparators used class balancing and separate Platt mappings fitted only to five-fold OOF training predictions.

## Fair raw and calibrated comparison

- Logistic regression: ROC-AUC **0.7491**, AP **0.2314**, Brier **0.2025 raw** and **0.0684 calibrated**.
- Explainable Boosting Machine: ROC-AUC **0.7686**, AP **0.2551**, Brier **0.1933 raw** and **0.0672 calibrated**.
- XGBoost: ROC-AUC **0.7687**, AP **0.2588**, Brier **0.1797 raw** and **0.0671 calibrated**.
- XGBoost-minus-logistic discrimination increments: ROC-AUC **0.0196** and AP **0.0274**.
- EBM-minus-logistic discrimination increments: ROC-AUC **0.0196** and AP **0.0237**.
- Any governance claim must weigh these increments against intrinsic interpretability and expected cost; ranking performance alone does not select an operational model.

## Expected cost per applicant

- FN:FP **1:1**: lowest locked-test cost = **0.0801** for XGBoost; margin to the next model = **0.0005** cost units per applicant.
- FN:FP **2:1**: lowest locked-test cost = **0.1548** for XGBoost; margin to the next model = **0.0011** cost units per applicant.
- FN:FP **5:1**: lowest locked-test cost = **0.3307** for Explainable Boosting Machine; margin to the next model = **0.0018** cost units per applicant.
- FN:FP **10:1**: lowest locked-test cost = **0.5162** for XGBoost; margin to the next model = **0.0005** cost units per applicant.
- FN:FP **20:1**: lowest locked-test cost = **0.6906** for Explainable Boosting Machine; margin to the next model = **0.0081** cost units per applicant.
- These are illustrative relative cost weights, not observed Vietnamese monetary losses or a commercially optimal policy.

## Paired test-set uncertainty

- XGBoost minus Logistic regression, roc_auc: difference **0.0196** (95% paired bootstrap CI **0.0163 to 0.0230**).
- XGBoost minus Logistic regression, pr_auc_average_precision: difference **0.0274** (95% paired bootstrap CI **0.0211 to 0.0339**).
- Explainable Boosting Machine minus Logistic regression, roc_auc: difference **0.0196** (95% paired bootstrap CI **0.0167 to 0.0224**).
- Explainable Boosting Machine minus Logistic regression, pr_auc_average_precision: difference **0.0237** (95% paired bootstrap CI **0.0190 to 0.0287**).
- XGBoost minus Explainable Boosting Machine, roc_auc: difference **0.0000** (95% paired bootstrap CI **-0.0022 to 0.0022**).
- XGBoost minus Explainable Boosting Machine, pr_auc_average_precision: difference **0.0037** (95% paired bootstrap CI **-0.0007 to 0.0081**).

## Selected XGBoost bootstrap uncertainty

- roc_auc: **0.7687** (95% bootstrap CI **0.7623 to 0.7752**).
- pr_auc_average_precision: **0.2588** (95% bootstrap CI **0.2484 to 0.2698**).
- precision: **0.2662** (95% bootstrap CI **0.2575 to 0.2749**).
- recall_sensitivity: **0.3925** (95% bootstrap CI **0.3795 to 0.4070**).
- specificity: **0.9050** (95% bootstrap CI **0.9027 to 0.9075**).
- f1: **0.3173** (95% bootstrap CI **0.3073 to 0.3276**).
- balanced_accuracy: **0.6488** (95% bootstrap CI **0.6421 to 0.6559**).
- mcc: **0.2503** (95% bootstrap CI **0.2391 to 0.2620**).
- brier_score: **0.0671** (95% bootstrap CI **0.0666 to 0.0675**).

## Split sensitivity (not temporal validation)

- XGBoost-minus-logistic roc_auc: mean **0.0175**, SD **0.0018**, range **0.0146 to 0.0216**, favourable in **100.0%** of 30 splits.
- XGBoost-minus-logistic pr_auc_average_precision: mean **0.0242**, SD **0.0036**, range **0.0166 to 0.0302**, favourable in **100.0%** of 30 splits.
- XGBoost-minus-logistic brier_score: mean **-0.0011**, SD **0.0001**, range **-0.0014 to -0.0008**, favourable in **100.0%** of 30 splits.
- XGBoost-minus-logistic log_loss: mean **-0.0052**, SD **0.0006**, range **-0.0063 to -0.0040**, favourable in **100.0%** of 30 splits.
- XGBoost minus Logistic regression, expected cost at FN:FP=1:1: mean difference **-0.0003** per applicant (range **-0.0006 to 0.0003**); the first model had lower cost in **86.7%** of splits.
- XGBoost minus Logistic regression, expected cost at FN:FP=2:1: mean difference **-0.0026** per applicant (range **-0.0040 to -0.0015**); the first model had lower cost in **100.0%** of splits.
- XGBoost minus Logistic regression, expected cost at FN:FP=5:1: mean difference **-0.0101** per applicant (range **-0.0157 to -0.0045**); the first model had lower cost in **100.0%** of splits.
- XGBoost minus Logistic regression, expected cost at FN:FP=10:1: mean difference **-0.0250** per applicant (range **-0.0339 to -0.0174**); the first model had lower cost in **100.0%** of splits.
- XGBoost minus Logistic regression, expected cost at FN:FP=20:1: mean difference **-0.0321** per applicant (range **-0.0410 to -0.0172**); the first model had lower cost in **100.0%** of splits.
- Explainable Boosting Machine minus Logistic regression, expected cost at FN:FP=1:1: mean difference **-0.0002** per applicant (range **-0.0006 to 0.0001**); the first model had lower cost in **73.3%** of splits.
- Explainable Boosting Machine minus Logistic regression, expected cost at FN:FP=2:1: mean difference **-0.0023** per applicant (range **-0.0039 to -0.0007**); the first model had lower cost in **100.0%** of splits.
- Explainable Boosting Machine minus Logistic regression, expected cost at FN:FP=5:1: mean difference **-0.0102** per applicant (range **-0.0142 to -0.0068**); the first model had lower cost in **100.0%** of splits.
- Explainable Boosting Machine minus Logistic regression, expected cost at FN:FP=10:1: mean difference **-0.0234** per applicant (range **-0.0301 to -0.0160**); the first model had lower cost in **100.0%** of splits.
- Explainable Boosting Machine minus Logistic regression, expected cost at FN:FP=20:1: mean difference **-0.0337** per applicant (range **-0.0430 to -0.0205**); the first model had lower cost in **100.0%** of splits.
- Explainable Boosting Machine threshold at FN:FP=5:1: mean **0.1599**, SD **0.0048**, range **0.1502 to 0.1679**.
- Logistic regression threshold at FN:FP=5:1: mean **0.1634**, SD **0.0032**, range **0.1549 to 0.1708**.
- XGBoost threshold at FN:FP=5:1: mean **0.1673**, SD **0.0059**, range **0.1517 to 0.1787**.
- roc_auc_gap_full_minus_without_ext: mean **0.0508**, SD **0.0025**, range **0.0464 to 0.0564**.
- pr_auc_gap_full_minus_without_ext: mean **0.0578**, SD **0.0038**, range **0.0509 to 0.0641**.
- application_train.csv has no usable application dates. Repeated random splits test partition sensitivity only and do not substitute for temporal or external validation.

## EXT_SOURCE ablation on the primary split

- Full XGBoost: ROC-AUC **0.7687**; AP **0.2588**.
- XGBoost without EXT_SOURCE_1/2/3: ROC-AUC **0.7196**; AP **0.1947**.
- XGBoost using EXT_SOURCE columns only: ROC-AUC **0.7208**; AP **0.2069**.

## PoC stress scenarios

- A: higher external scores / lower credit-income ratio: N=300, median calibrated XGBoost score **0.013**, IQR **0.009-0.018**.
- B: EXT_SOURCE_1 missing / moderate other scores: N=300, median calibrated XGBoost score **0.034**, IQR **0.022-0.052**.
- C: lower external scores / higher credit-income ratio: N=300, median calibrated XGBoost score **0.126**, IQR **0.086-0.181**.

## Interpretation and deployment limits

- EBM global and local contributions are intrinsic additive effects expressed on the raw default log-odds scale. Platt calibration is a separate probability mapping.
- TreeSHAP remains a post-hoc description of the fitted XGBoost model; it does not establish causality, legitimacy, fairness or regulatory compliance.
- Home Credit is a public technical proxy whose country/platform provenance is undisclosed; external validity for Vietnam is not established.
- EXT_SOURCE_1/2/3 are undocumented external scores and cannot be relabelled as CIC or digital-footprint variables.
- The PoC is a presentation-only sensitivity demonstration. It must load a locked artifact and must not be described as a live Vietnamese underwriting or automated approval system.
