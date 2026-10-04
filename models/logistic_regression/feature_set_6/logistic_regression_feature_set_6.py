"""Regional/wealth-adjusted R-score + work-hours trial; needs numpy and pandas.

Historical labels identify conditional associations, not hidden qualification.
Run this file from any directory. Outputs stay alongside it.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
STEM = "logistic_regression_feature_set_6"
NUMERIC = ["cote_r_equivalent", "revenu_familial_estime", "heures_travail_semaine"]
REMOTE = {"Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine"}
CENTRE = {"Montreal", "Capitale-Nationale"}
SEED, BUDGET, EO_TOL, L2 = 42, 0.40, 0.01, 1.0


def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -np.asarray(z)))


def remote_mask(frame):
    assert set(frame.region_administrative) <= REMOTE | CENTRE
    return frame.region_administrative.isin(REMOTE).to_numpy()


class LogisticModel:
    """Standardized logistic regression, Newton solver with backtracking."""

    def __init__(self, variant="adjusted"):
        self.variant = variant

    def design(self, frame):
        z = (frame[NUMERIC].to_numpy(float) - self.mean) / self.scale
        remote = remote_mask(frame).astype(float)
        parts = [np.ones(len(frame)), z[:, 0]]
        self.names = ["intercept", "R_z"]
        if self.variant != "r_only":
            parts.append(z[:, 2])
            self.names.append("hours_z")
        if self.variant.startswith("adjusted"):
            parts.extend([z[:, 1], remote])
            self.names.extend(["income_z", "remote"])
        if self.variant == "adjusted_interactions":
            parts.extend([remote * z[:, j] for j in range(3)])
            self.names.extend(["remote_x_R_z", "remote_x_income_z", "remote_x_hours_z"])
        if self.variant == "adjusted_extras":
            parts.append(frame.premiere_generation_universitaire.to_numpy(float))
            self.names.append("first_generation")
            for program in self.programs[1:]:
                parts.append((frame.programme_etudes == program).to_numpy(float))
                self.names.append("program_" + program)
        return np.column_stack(parts)

    def fit(self, frame, y):
        a = frame[NUMERIC].to_numpy(float)
        self.mean, self.scale = a.mean(0), a.std(0)
        assert np.all(self.scale > 0)
        self.programs = sorted(frame.programme_etudes.unique())
        x = self.design(frame)
        y = np.asarray(y, float)
        penalty = np.full(x.shape[1], L2)
        penalty[0] = 0
        beta = np.zeros(x.shape[1])

        def loss(b):
            z = x @ b
            return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(penalty * b * b)

        for iteration in range(100):
            p = sigmoid(x @ beta)
            gradient = x.T @ (p - y) + penalty * beta
            if np.max(np.abs(gradient)) < 1e-6:
                break
            hessian = (x.T * (p * (1 - p))) @ x + np.diag(penalty)
            step = np.linalg.solve(hessian, gradient)
            alpha, previous = 1.0, loss(beta)
            while loss(beta - alpha * step) > previous - 1e-4 * alpha * (gradient @ step):
                alpha *= 0.5
                if alpha < 1e-12:
                    raise RuntimeError("Logistic solver line search failed")
            beta -= alpha * step
        else:
            raise RuntimeError("Logistic solver did not converge")
        self.beta = beta
        self.gradient_max = float(np.max(np.abs(gradient)))
        return self

    def predict(self, frame):
        return sigmoid(self.design(frame) @ self.beta)

    def qualification(self, frame, reference_income):
        # This intervention defines our trial assumption, not a causal claim.
        neutral = frame.copy()
        neutral["revenu_familial_estime"] = reference_income
        neutral["region_administrative"] = "Montreal"
        return self.predict(neutral)

    def raw_coefficients(self):
        b = dict(zip(self.names, self.beta))
        return {
            "R_per_point": float(b["R_z"] / self.scale[0]),
            "hours_per_hour": float(b.get("hours_z", 0) / self.scale[2]),
            "income_per_10000": float(b.get("income_z", 0) * 10000 / self.scale[1]),
            "remote_log_odds": float(b.get("remote", 0)),
        }


def metrics(y, p):
    y = np.asarray(y)
    ranks = pd.Series(p).rank(method="average").to_numpy()
    n1, n0 = int(y.sum()), int((1 - y).sum())
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return {"historical_log_loss": float(-np.mean(y * np.log(p) + (1-y) * np.log(1-p))),
            "historical_auc": float(auc), "historical_accuracy": float(np.mean((p >= .5) == y))}


def folds_for(frame):
    rng = np.random.default_rng(SEED)
    strata = 2 * remote_mask(frame).astype(int) + frame.decision_octroi.to_numpy()
    folds = np.empty(len(frame), int)
    for value in np.unique(strata):
        indices = np.flatnonzero(strata == value)
        rng.shuffle(indices)
        folds[indices] = np.arange(len(indices)) % 5
    return folds


def allocate(q, remote, ids, budget=BUDGET, tolerance=EO_TOL):
    """Joint exact budget + soft EO; q is an assumed qualification probability.

    Soft TPR_g = sum(q_i * selected_i)/sum(q_i) within group g.
    Enumerate group grant counts, NOT every applicant combination. Within each
    group, the highest q maximizes expected agreement at a fixed grant count.
    """
    k = round(budget * len(q))
    indices = [np.flatnonzero(~remote), np.flatnonzero(remote)]
    ordered = [idx[np.lexsort((ids[idx], -q[idx]))] for idx in indices]
    prefixes = [np.r_[0., np.cumsum(q[idx])] for idx in ordered]
    centre_k = np.arange(max(0, k - len(ordered[1])), min(k, len(ordered[0])) + 1)
    remote_k = k - centre_k
    tpr_c = prefixes[0][centre_k] / prefixes[0][-1]
    tpr_r = prefixes[1][remote_k] / prefixes[1][-1]
    gap = np.abs(tpr_c - tpr_r)
    selected_mass = prefixes[0][centre_k] + prefixes[1][remote_k]
    feasible = np.flatnonzero(gap <= tolerance + 1e-12)
    if not len(feasible):
        raise RuntimeError(f"No feasible allocation at budget={budget}, tolerance={tolerance}")
    j = feasible[np.lexsort((centre_k[feasible], gap[feasible], -selected_mass[feasible]))[0]]
    pred = np.zeros(len(q), int)
    pred[ordered[0][:centre_k[j]]] = 1
    pred[ordered[1][:remote_k[j]]] = 1
    record = {
        "budget": float(pred.mean()), "eo_tolerance": float(tolerance),
        "soft_eo_gap": float(gap[j]), "soft_tpr_centre": float(tpr_c[j]),
        "soft_tpr_remote": float(tpr_r[j]),
        "centre_grant_rate": float(pred[~remote].mean()),
        "remote_grant_rate": float(pred[remote].mean()),
        "demographic_parity_gap": float(abs(pred[~remote].mean() - pred[remote].mean())),
        "expected_agreement_under_q": float(np.mean(pred*q + (1-pred)*(1-q))),
    }
    assert pred.sum() == k and 0.36 <= pred.mean() <= 0.44
    return pred, record


def main():
    data = ROOT / "equialgo-participants" / "data"
    history = pd.read_csv(data / "donnees_demandes.csv")
    evaluation = pd.read_csv(data / "candidats_evaluation.csv")
    assert len(history) == 10000 and len(evaluation) == 4000
    assert history.id_candidat.is_unique and evaluation.id_candidat.is_unique
    assert not set(history.id_candidat) & set(evaluation.id_candidat)
    assert not history.isna().any().any() and not evaluation.isna().any().any()
    y = history.decision_octroi.to_numpy()
    fold_id = folds_for(history)
    cv_rows, coefficients = [], []
    for variant in ["r_only", "r_hours", "adjusted", "adjusted_interactions", "adjusted_extras"]:
        oof = np.empty(len(history))
        for fold in range(5):
            train, valid = fold_id != fold, fold_id == fold
            model = LogisticModel(variant).fit(history.loc[train], y[train])
            oof[valid] = model.predict(history.loc[valid])
            if variant == "adjusted":
                coefficients.append({"fold": fold, **model.raw_coefficients()})
        cv_rows.append({"variant": variant, **metrics(y, oof)})
    pd.DataFrame(cv_rows).to_csv(OUT / f"{STEM}_historical_cv.csv", index=False)
    pd.DataFrame(coefficients).to_csv(OUT / f"{STEM}_coefficient_stability.csv", index=False)
    print("Historical-label diagnostics ONLY; these do not measure hidden accuracy:")
    print(pd.DataFrame(cv_rows).round(5).to_string(index=False), flush=True)

    model = LogisticModel().fit(history, y)
    reference_income = float(history.revenu_familial_estime.median())
    q = model.qualification(evaluation, reference_income)
    remote = remote_mask(evaluation)
    ids = evaluation.id_candidat.to_numpy()
    pred, chosen = allocate(q, remote, ids)
    plain = np.zeros(len(q), int)
    plain[np.lexsort((ids, -q))[:round(BUDGET * len(q))]] = 1
    # A soft EO target is usually feasible only up to finite-count precision.
    frontier = []
    for tol in [.002, .005, .01, .02, .05, .10, 1.]:
        _, row = allocate(q, remote, ids, tolerance=tol)
        frontier.append(row)
    pd.DataFrame(frontier).to_csv(OUT / f"{STEM}_pareto.csv", index=False)

    # Check sensitivity to the reference income and calibration assumption.
    sensitivity = []
    for quantile in [.25, .5, .75]:
        income = float(history.revenu_familial_estime.quantile(quantile))
        qq = model.qualification(evaluation, income)
        alternative, row = allocate(qq, remote, ids)
        sensitivity.append({"reference_income_quantile": quantile,
                            "reference_income": income,
                            "changed_vs_primary": int((alternative != pred).sum()), **row})

    baseline_path = OUT.parent / "feature_set_1" / "logistic_regression_feature_set_1_predictions.csv"
    baseline = pd.read_csv(baseline_path).set_index("id_candidat").reindex(ids)
    assert not baseline.decision_octroi.isna().any()
    old = baseline.decision_octroi.to_numpy(int)
    output = pd.DataFrame({"id_candidat": ids, "decision_octroi": pred})
    output_path = OUT / f"{STEM}_predictions.csv"
    output.to_csv(output_path, index=False)
    check = pd.read_csv(output_path)
    assert check.equals(output) and check.id_candidat.equals(evaluation.id_candidat)
    assert set(check.decision_octroi) == {0, 1} and check.decision_octroi.sum() == 1600

    score_table = evaluation[["id_candidat", "region_administrative", *NUMERIC]].copy()
    score_table["qualification_estimate"] = q
    score_table["decision_octroi"] = pred
    score_table["baseline_decision"] = old
    score_table["changed_vs_baseline"] = pred != old
    score_table.to_csv(OUT / f"{STEM}_scores.csv", index=False)
    report = {
        "seed": SEED, "l2_penalty": L2, "reference_income": reference_income,
        "training_features": NUMERIC + ["remote_indicator"],
        "ranking_features": ["cote_r_equivalent", "heures_travail_semaine"],
        "raw_coefficients": model.raw_coefficients(), "solver_gradient_max": model.gradient_max,
        "equivalent_R_score_per_work_hour": model.raw_coefficients()["hours_per_hour"] / model.raw_coefficients()["R_per_point"],
        "fairness_adjustment_changes_vs_global_top_k": int((plain != pred).sum()),
        "allocation": chosen, "historical_cv": cv_rows,
        "coefficient_stability": coefficients, "reference_income_sensitivity": sensitivity,
        "changed_vs_baseline": int((pred != old).sum()),
        "added_vs_baseline": int(((pred == 1) & (old == 0)).sum()),
        "removed_vs_baseline": int(((pred == 0) & (old == 1)).sum()),
        "baseline_centre_grant_rate": float(old[~remote].mean()),
        "baseline_remote_grant_rate": float(old[remote].mean()),
        "hidden_reference_accuracy": None, "hidden_reference_equal_opportunity": None,
        "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in [data / "donnees_demandes.csv", data / "candidats_evaluation.csv",
                                baseline_path, output_path]},
        "limitations": [
            "Regional adjustment assumes a shared additive score with a regional intercept penalty.",
            "Holding income fixed removes its direct historical preference, not every income proxy.",
            "Positive work-hour effect is assumed legitimate; hidden qualification may differ.",
            "Soft EO and expected agreement use model-estimated qualification, not hidden labels.",
            "Historical-label CV diagnoses the committee model and cannot establish leaderboard improvement.",
        ],
    }
    (OUT / f"{STEM}_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"coefficients": report["raw_coefficients"], "allocation": chosen,
                      "changed_vs_baseline": report["changed_vs_baseline"],
                      "reference_income_sensitivity": sensitivity}, indent=2))
    print(f"Validated submission: {output_path}")


if __name__ == "__main__":
    main()
