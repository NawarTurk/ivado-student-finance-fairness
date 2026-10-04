# Prompt 1

Create `qualification.ipynb`. Goal: train a centre-only qualification model and
inspect it. Do NOT produce predictions.csv or set any thresholds yet.

1. Load data/donnees_demandes.csv and data/candidats_evaluation.csv.
   Add a `groupe` column ('Centre' / 'Eloignee') to both using the same
   logic as groupe_region in baseline_model_en.ipynb.

2. Feature set: cote_r_equivalent, revenu_familial_estime,
   heures_travail_semaine, premiere_generation_universitaire,
   programme_etudes (one-hot).
   EXCLUDE region_administrative, code_postal_3, distance_domicile_campus_km.

3. Subset to Centre rows only. Split 80/20 stratified on decision_octroi,
   random_state=42. Train a gradient boosting classifier (sklearn
   HistGradientBoostingClassifier, random_state=42) on the training part.

4. Report on the held-out centre rows:
   - accuracy, AUC
   - a calibration check: bin predicted probability into deciles and print
     predicted vs observed grant rate per bin

5. Score ALL applicants (both groups, both files) with this model. Call the
   output `q`. Use predict_proba, not predict.

6. Diagnostics, printed:
   a. mean q by groupe, for the historical data, next to the actual
   decision_octroi rate by groupe
   b. for R-score bins [15,25,27,29,31,40] on historical data: a table with
   per-group counts, mean decision_octroi, and mean q
   c. mean q by groupe on the 4,000 evaluation applicants
   d. permutation importance of each feature on the held-out centre rows

7. Save the historical data with the q column to `data/q_historique.csv` and
   the evaluation data with q to `data/q_evaluation.csv`.

Run the notebook end to end and summarize the printed numbers. Flag anything
that looks inconsistent with the premise that q is less regionally biased than
decision_octroi.

# Prompt 2

Create `model_corrige.ipynb`. Goal: a valid predictions.csv using per-group
thresholds. One fixed setting only; the Pareto sweep comes later.

1. Load data/q_historique.csv and data/q_evaluation.csv.

2. Build the reference label: `qualifie` = 1 for the top 40% of q overall
   (single global cutoff, both groups pooled), on BOTH files. Print the
   resulting qualifie rate by groupe for each file.

3. Scoring model: retrain the baseline RandomForestClassifier exactly as in
   baseline_model_en.ipynb (all columns including region and postal code,
   n_estimators=300, min_samples_leaf=20, random_state=42) on all 10,000
   historical rows with decision_octroi as the label. Use predict_proba on
   the 4,000 evaluation applicants to get a score s.

4. Per-group thresholds on the evaluation set:
   - For each group, define TPR(t) = share of qualifie applicants in that
     group with s >= t.
   - Search over a grid of (t_centre, t_eloignee) pairs. Keep pairs where
     |TPR_centre - TPR_eloignee| <= 0.01 AND the overall grant rate is
     between 0.38 and 0.42.
   - Among those, pick the pair maximizing the number of qualifie applicants
     selected. Print the chosen thresholds, per-group TPR, per-group grant
     rate, and overall grant rate.
   - If no pair satisfies both, STOP and report the closest achievable
     combinations. Do not relax the budget.

5. Write predictions.csv at the repo root: columns id_candidat,
   decision_octroi; 4,000 rows; integers 0/1.

6. Final verification, printed:
   - row count, column names, unique values in decision_octroi
   - overall grant rate with an explicit OK / OUT OF BUDGET check
   - grant rate by groupe
   - TPR by groupe against qualifie, and the EO gap

Run end to end and summarize.
