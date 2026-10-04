from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CENTER_REGIONS = {"Montreal", "Capitale-Nationale"}
REMOTE_REGIONS = {"Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine"}


def ensure_region_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add a simple center/remote label if it is not already present."""
    if "region_group" not in df.columns:
        df = df.copy()
        df["region_group"] = df["region_administrative"].apply(
            lambda region: (
                "center" if region in CENTER_REGIONS else "remote" if region in REMOTE_REGIONS else "other"
            )
        )
    return df


def compute_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the qualified-vs-granted confusion matrix with both axes in {0,1}."""
    matrix = pd.crosstab(
        df["qualified_proxy"],
        df["decision_octroi"],
        rownames=["qualified_proxy"],
        colnames=["granted"],
    )
    return matrix.reindex(index=[0, 1], columns=[0, 1], fill_value=0)


def audit_table(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """Summarize qualification and grant rates overall and by region group."""
    rows = {}
    groups = [("All", df), *df.groupby("region_group")]
    for name, group in groups:
        qualified = group["qualified_proxy"] == 1
        rows[name] = {
            "n": len(group),
            "% qualified": 100 * qualified.mean(),
            "% granted": 100 * group[pred_col].mean(),
            "% qualified who got it": 100 * group.loc[qualified, pred_col].mean(),
            "% unqualified who got it": 100 * group.loc[~qualified, pred_col].mean(),
        }
    return pd.DataFrame(rows).T.round(2)


def save_confusion_plot(matrix: pd.DataFrame, title: str, save_path: Path) -> None:
    """Save a confusion-matrix image from a 2x2 count table."""
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix.values, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    ax.set_xlabel("granted")
    ax.set_ylabel("qualified_proxy")
    ax.set_title(title)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix.iloc[row, col])
            text_color = "black" if value < (matrix.to_numpy().max() / 2) else "white"
            ax.text(col, row, str(value), ha="center", va="center", color=text_color)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_group_plot(df: pd.DataFrame, group_name: str, output_dir: Path) -> None:
    """Create the confusion-matrix PNG for a subset of rows."""
    subset = df if group_name == "all" else df[df["region_group"] == group_name].copy()
    matrix = compute_matrix(subset)
    title = f"Qualified vs Granted - {group_name.title()}"
    save_path = output_dir / f"confusion_matrix_{group_name}.png"
    save_confusion_plot(matrix, title, save_path)


def generate_all_confusion_plots() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data_with_quality.csv"
    output_dir = project_root
    output_dir.mkdir(exist_ok=True)

    df = ensure_region_group(pd.read_csv(data_path))
    table = audit_table(df, "decision_octroi")
    print(table)
    table_path = output_dir / "fairness_metrics.csv"
    table.to_csv(table_path, index=True, index_label="group")
    print(f"Saved: {table_path}")
    print()

    for group_name in ["all", "remote", "center"]:
        subset = df if group_name == "all" else df[df["region_group"] == group_name].copy()
        matrix = compute_matrix(subset)
        title = f"Qualified vs Granted - {group_name.title()}"
        save_path = output_dir / f"confusion_matrix_{group_name}.png"
        save_confusion_plot(matrix, title, save_path)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    generate_all_confusion_plots()
