# Feature set 6: R score and hours, adjusted for committee effects

Upload **`logistic_regression_feature_set_6_predictions.csv`**. It contains the
original 4,000 candidate IDs and binary decisions, with exactly 1,600 grants
(40%). Earlier feature sets and their prediction files are unchanged.

## Trial hypothesis

The committee's decisions can be approximated by a shared logistic score plus
a regional penalty. Train on R score, work hours, income, and a Centre/Remote
indicator. At prediction time hold income at the historical median ($61,834)
and remove the remote penalty. Thus **only R score and work hours determine
the qualification ranking**. Income and region are adjustment variables in
training. Region is also available for the final fairness constraint.

The fitted conditional log-odds effects are approximately:

| Variable | Effect |
|---|---:|
| R score, per point | +1.36169 |
| Work hours, per hour/week | +0.19555 |
| Income, per $10,000 | +0.23381 |
| Remote region | -1.96858 |

After neutralizing income and region, the ranking is equivalent to:

**R score + 0.14361 × work hours per week.**

These effects are associations with committee decisions, not recovered ground
truth. The trial assumes the positive work-hours effect is legitimate and
removes the direct preference for wealth. It does not claim to remove all
income proxies or establish a causal effect.

## Budget and fairness

Choose Centre and Remote grant counts jointly, keeping the highest qualification
scores within each group. Maximize expected agreement under the qualification
estimate while requiring exactly 40% grants and a soft equal-opportunity gap
of at most 0.01. Soft TPR is `sum(q * selected) / sum(q)` within each group.

This is **model-estimated fairness**, not equal opportunity measured against
the hidden reference. The q estimates derive from historical decisions after
the stated adjustments; they are not calibrated against hidden qualification.

For this dataset the unconstrained highest-score allocation already satisfies
the soft constraint, so the fairness step changes no decisions. Results:

- Centre: 949 grants / 2,372 applicants = 40.0084%.
- Remote: 651 grants / 1,628 applicants = 39.9877%.
- Demographic-parity gap: 0.0207 percentage points.
- Soft equal-opportunity gap: 0.3561 percentage points.
- Relative to feature set 1: 96 newly selected and 96 newly rejected candidates.
- Using the 25th or 75th income percentile instead of the median as the common
  reference leaves all final decisions unchanged.

## Diagnostics and limits

Five folds are stratified by committee label and Centre/Remote group. Scaling
is fit separately on training rows in each fold. Logistic coefficients use a
fixed L2 penalty of 1, with no hyperparameter search. The adjusted committee
model has out-of-fold historical log loss 0.26054 and AUC 0.95651. These numbers
describe committee prediction **before** the qualification adjustment; they
cannot be compared to the baseline's reported 92.53% leaderboard accuracy.

Regional interactions reduce historical log loss only to 0.26040. Adding
program and first-generation status increases it to 0.26129. Neither diagnostic
provides a compelling reason to expand this first trial. Fold coefficients
and a surrogate Pareto table are saved alongside the predictions.

**Hidden-reference accuracy, macro F1, and equal opportunity are unknown until
evaluation.** This CSV is a trial, not a demonstrated improvement over feature
set 1. Its main uncertainty is whether hours worked belongs in the hidden
qualification standard with the inferred weight.

## Reproduce

With Python, NumPy and pandas installed, run from this directory:

```sh
python3 logistic_regression_feature_set_6.py
```

The script also works from other working directories. It uses a small NumPy
Newton solver with convergence checks and a fixed seed (42). The notebook runs
the same script. The report records settings, diagnostics, and SHA-256 hashes
of the input datasets, baseline CSV and new CSV. The scores CSV is for auditing;
upload only the two-column predictions CSV.
