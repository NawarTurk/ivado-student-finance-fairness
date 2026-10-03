# ÉquiAlgo hackathon: project context

## Goal

Correct regional bias in student-financing decisions. Scored against a hidden
reference standard, NOT against `decision_octroi`.

## Key facts

- `decision_octroi` is the committee's historical decision and is BIASED against
  remote regions. Never treat it as ground truth for fairness evaluation.
- Remote = Bas-Saint-Laurent, Cote-Nord, Gaspesie-Iles-de-la-Madeleine.
  Use `groupe_region()` from baseline_model_en.ipynb.
- Columns: cote_r_equivalent, programme_etudes, region_administrative,
  code_postal_3, revenu_familial_estime, heures_travail_semaine,
  distance_domicile_campus_km, premiere_generation_universitaire, decision_octroi.
- Evidence of bias: within every R-score bin, remote applicants are granted less
  often than centre applicants at the same score (45% vs 17% at R 27-29,
  89% vs 65% at 29-31). Acceptance is flat across distance bins within each
  group, so distance carries region and not merit.

## Approach

Train a qualification model on CENTRE applicants only, using decision_octroi as
the label, excluding all regional features. The assumption is that centre
decisions reflect the standard without the remote penalty. Apply that model to
all applicants to get a qualification score `q`.

## Design decisions (do not change without asking)

- Excluded from the qualification model: region_administrative, code_postal_3,
  distance_domicile_campus_km. These carry region, not merit.
- Fairness metric: equal opportunity, measured against OUR qualification
  estimate, never against decision_octroi.
- Final grant rate on candidats_evaluation.csv must be 36-44%.
- Per-group thresholds, set later, are what enforce equal opportunity.
  A single global threshold is not the plan.

## Working rules

- Don't modify baseline_model_en.ipynb. New work goes in new files.
- Fix random seeds (42).
- Don't optimize for accuracy against decision_octroi. It is a biased label.
- If a method fails or a result looks off, STOP and report it. Don't silently
  substitute a different model, add an excluded feature, or tune to make a
  number look better.
- Print the numbers listed in each task so results can be checked without
  reading all the code.\*\*\*\*
