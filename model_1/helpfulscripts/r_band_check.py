from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from print_quality_data import build_proxy_quality
except ImportError:
    from proxy_quality import build_proxy_quality


def main() -> None:
    model_dir = Path(__file__).resolve().parent.parent
    raw_path = model_dir.parent / "equialgo-participants" / "data" / "donnees_demandes.csv"

    df = pd.read_csv(raw_path)
    out = build_proxy_quality(df, save_path=None)
    band_labels = ["1 lowest", "2", "3", "4", "5 highest"]
    out["r_band"] = pd.qcut(out["cote_r_equivalent"], q=5, labels=band_labels)

    grouped = out.pivot_table(
        index="r_band",
        columns="region_group",
        values="decision_octroi",
        aggfunc=["mean", "count"],
        observed=True,
    )
    rates = grouped["mean"].reindex(columns=["center", "remote"]) * 100
    counts = grouped["count"].reindex(columns=["center", "remote"])

    table = pd.DataFrame(index=rates.index)
    table["center"] = rates["center"]
    table["remote"] = rates["remote"]
    table["gap (center - remote)"] = table["center"] - table["remote"]
    table["n_center"] = counts["center"]
    table["n_remote"] = counts["remote"]

    print(table.round(1))
    table.round(2).to_csv(model_dir / "r_band_grant_rate.csv", index=True, index_label="r_band")

    ax = table[["center", "remote"]].plot(kind="bar", figsize=(9, 5))
    ax.set_title("Scholarship Grant Rate by R-Score Band")
    ax.set_xlabel("R-score band")
    ax.set_ylabel("Grant rate (%)")
    ax.legend(title="Region group")
    ax.tick_params(axis="x", rotation=0)
    ax.figure.tight_layout()
    ax.figure.savefig(model_dir / "r_band_grant_rate.png", dpi=150)
    plt.close(ax.figure)

    if (table["gap (center - remote)"] > 0).all():
        print("Remote grant rates are lower than center in every R-score band.")
    else:
        print("Remote grant rates are not lower than center in every R-score band.")


if __name__ == "__main__":
    main()
