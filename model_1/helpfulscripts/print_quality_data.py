from pathlib import Path

import pandas as pd

CENTER_REGIONS = {"Montreal", "Capitale-Nationale"}
REMOTE_REGIONS = {"Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine"}


def print_quality_preview(path: str | None = None, rows: int = 10) -> None:
    """Print only the first rows of the full enriched dataset."""
    if path is None:
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / "data_with_quality.csv"
    else:
        path = Path(path)

    df = pd.read_csv(path)

    print(f"Loaded dataset: {path}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"\nFirst {rows} rows of data_with_quality.csv:")
    print(df.head(rows).to_string(index=False))


if __name__ == "__main__":
    print_quality_preview()
