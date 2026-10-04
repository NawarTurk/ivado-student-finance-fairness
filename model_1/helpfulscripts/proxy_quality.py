from pathlib import Path

import pandas as pd

CENTER_REGIONS = {"Montreal", "Capitale-Nationale"}
REMOTE_REGIONS = {"Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine"}


def minmax_scale(series: pd.Series) -> pd.Series:
    """Scale a numeric series to [0, 1]."""
    smin = series.min()
    smax = series.max()
    if smax == smin:
        return pd.Series(0.5, index=series.index)
    return (series - smin) / (smax - smin)


def build_proxy_quality(
    df: pd.DataFrame,
    save_path: str | None = ".",
    save_name: str = "data_with_quality.csv",
) -> pd.DataFrame:
    """
    Build a proxy quality score from merit and need signals.

    The score is intentionally a proxy, not a true ground-truth label.
    It is meant for fairness analysis, score-banding, and explanation.

    Parameters:
        df: input dataframe
        save_path: optional directory path to save the output csv
        save_name: filename for the saved output file
    """
    out = df.copy()

    # Higher R-score is better
    out["score_r"] = minmax_scale(out["cote_r_equivalent"])

    # Lower income is better for need, so invert the minmax scaling
    out["score_income"] = 1 - minmax_scale(out["revenu_familial_estime"])

    # More hours worked is treated as higher burden / need
    out["score_hours"] = minmax_scale(out["heures_travail_semaine"])

    # Weighted composite score inside [0, 1]
    # R-score gets weight 3; income and hours each get weight 0.5
    out["proxy_quality_score"] = (
        3 * out["score_r"] + 0.5 * out["score_income"] + 0.5 * out["score_hours"]
    ) / 4

    # Qualify approximately the top 40% of applicants by proxy score
    cut = out["proxy_quality_score"].quantile(0.60)
    out["qualified_proxy"] = (out["proxy_quality_score"] >= cut).astype(int)

    # Derived fairness grouping used in the audit: center vs remote
    out["region_group"] = out["region_administrative"].apply(
        lambda region: (
            "center"
            if region in CENTER_REGIONS
            else "remote" if region in REMOTE_REGIONS else "other"
        )
    )

    if save_path is not None:
        save_target = f"{save_path.rstrip('/')}/{save_name}"
        out.to_csv(save_target, index=False)

    return out


def generate_quality_dataset() -> pd.DataFrame:
    """Read the repository dataset and save the enriched CSV under model_1."""
    model_dir = Path(__file__).resolve().parent.parent
    repository_root = model_dir.parent
    raw_path = repository_root / "equialgo-participants" / "data" / "donnees_demandes.csv"
    out = build_proxy_quality(
        pd.read_csv(raw_path),
        save_path=str(model_dir),
        save_name="data_with_quality.csv",
    )
    print(f"Created dataset: {model_dir / 'data_with_quality.csv'}")
    print("Qualified share by region:")
    print(out.groupby("region_group")["qualified_proxy"].mean())
    return out


def save_quality_summary(
    df: pd.DataFrame,
    save_path: str = ".",
    save_name: str = "quality_summary.csv",
) -> pd.DataFrame:
    """Create and save a compact summary of the quality-scored dataset."""
    if "proxy_quality_score" not in df.columns:
        raise ValueError("The dataframe must contain 'proxy_quality_score'.")

    region_summary = (
        df.groupby("region_administrative", dropna=False)
        .agg(
            applicants=("id_candidat", "count"),
            avg_quality=("proxy_quality_score", "mean"),
            qualified_share=("qualified_proxy", "mean"),
            avg_r_score=("cote_r_equivalent", "mean"),
            avg_income=("revenu_familial_estime", "mean"),
            avg_hours=("heures_travail_semaine", "mean"),
        )
        .reset_index()
    )

    overall_summary = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "mean_quality_score",
                "min_quality_score",
                "max_quality_score",
                "qualified_share",
            ],
            "value": [
                len(df),
                len(df.columns),
                df["proxy_quality_score"].mean(),
                df["proxy_quality_score"].min(),
                df["proxy_quality_score"].max(),
                df["qualified_proxy"].mean(),
            ],
        }
    )

    summary = {
        "overall_summary": overall_summary,
        "region_summary": region_summary,
    }

    save_target = f"{save_path.rstrip('/')}/{save_name}"
    overall_summary.to_csv(save_target, index=False)

    by_region_path = f"{save_path.rstrip('/')}/quality_summary_by_region.csv"
    region_summary.to_csv(by_region_path, index=False)

    return summary


if __name__ == "__main__":
    generate_quality_dataset()
