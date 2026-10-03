import pandas as pd

from model_1.helpfulscripts.proxy_quality import build_proxy_quality, save_quality_summary


if __name__ == "__main__":
    input_path = "equialgo-participants/data/donnees_demandes.csv"
    output_path = "."

    df = pd.read_csv(input_path)
    df = build_proxy_quality(df, save_path=output_path)
    summary = save_quality_summary(df, save_path=output_path)

    print("Saved enriched dataset to:", output_path + "/data_with_quality.csv")
    print("Saved quality summary to:", output_path + "/quality_summary.csv")
    print(df[["proxy_quality_score", "qualified_proxy"]].head())
    print("\nOverall summary:")
    print(summary["overall_summary"].to_string(index=False))
