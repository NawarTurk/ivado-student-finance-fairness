from pathlib import Path

import pandas as pd

if __package__:
    from .helpfulscripts.proxy_quality import build_proxy_quality, save_quality_summary
else:
    from helpfulscripts.proxy_quality import build_proxy_quality, save_quality_summary


if __name__ == "__main__":
    model_dir = Path(__file__).resolve().parent
    repository_root = model_dir.parent
    input_path = repository_root / "equialgo-participants" / "data" / "donnees_demandes.csv"

    df = pd.read_csv(input_path)
    df = build_proxy_quality(df, save_path=str(model_dir))
    summary = save_quality_summary(df, save_path=str(model_dir))

    print("Saved enriched dataset to:", model_dir / "data_with_quality.csv")
    print("Saved quality summary to:", model_dir / "quality_summary.csv")
    print(df[["proxy_quality_score", "qualified_proxy"]].head())
    print("\nOverall summary:")
    print(summary["overall_summary"].to_string(index=False))
