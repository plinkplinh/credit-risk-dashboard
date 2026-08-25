# Survey analysis audit summary

## Sample flow

- Google Forms submissions: **318**.
- Consented: **293**; declined: **25**.
- Primary analytical sample: **N=293**.
- Non-consenting submissions were excluded before any substantive analysis.

## Data-integrity audit

- Source CSV SHA-256: **`3d6d6205400a458997a9ed06716ccd218ed6025ec9608ead97a26609bc0d39cc`**; reference-hash match: **True**.
- Raw duplicate rows: **0**; repeated timestamps: **0**.
- Identical answer profiles: **4 rows in 2 pairs/groups**. These are not treated as proof of duplicate people.
- Flagged interval: **243 raw submissions**, including **221 consenting responses**; **72 consenting responses** lie outside it.
- Refusal selected together with another motivation: **107** responses. This is reported as a questionnaire inconsistency flag, not silently deleted.
- All four Likert answers identical: **15** responses. Four equal closed-item ratings alone are insufficient evidence of careless responding.

## Corrected headline estimates

- No self-reported CIC history: **155/293 (52.9%)**.
- Traditional commercial banks most trusted: **108/293 (36.9%)**.
- Absolute refusal regardless of interest discount: **64/293 (21.8%)**.
- Refusal regardless of any listed benefit: **125/293 (42.7%)**.
- Multi-select percentages are respondent prevalence and may sum to more than 100%.

## Technology-acceptance items

- Open Banking/API knowledge: median **3**, agree/strongly agree **132/293 (45.1%)**.
- Permission-management confidence: median **3**, agree/strongly agree **134/293 (45.7%)**.
- QR/bank-statement willingness: median **3**, agree/strongly agree **118/293 (40.3%)**.
- E-commerce/e-wallet statement willingness: median **3**, agree/strongly agree **125/293 (42.7%)**.

## Inferential-analysis rules

- 12 exploratory comparisons were run; p-values were adjusted using Benjamini-Hochberg false-discovery-rate control.
- Effect sizes and assumption diagnostics are reported alongside p-values.
- Non-significance is not interpreted as evidence that groups are identical.
- These tests are exploratory, not preregistered confirmatory hypothesis tests.

## Construct limitation

- PEU proxy: Spearman-Brown reliability **0.363**. Low: report items separately; do not claim a validated composite scale.
- PU proxy: Spearman-Brown reliability **0.403**. Low: report items separately; do not claim a validated composite scale.

## Required interpretation limits

- Convenience and snowball recruitment do not provide a probability sample or a calculable response rate.
- Results are exploratory and must not be generalised to all Vietnamese consumers.
- The burst sensitivity analysis is mandatory because recruitment-wave composition differs materially.
- The survey measures self-reported CIC history and stated willingness, not verified credit records or observed data-sharing behaviour.
- Survey records are not linked to Home Credit records and do not validate the XGBoost model on Vietnamese borrowers.
- Completion time cannot be screened because the file contains submission time only, not start time or duration.
- The two-item PU and PEU proxies are not validated TAM scales; analyse the four items separately.
