from pathlib import Path

import pandas as pd


def main() -> None:
    model_dir = Path(__file__).resolve().parent.parent
    raw_path = model_dir.parent / "equialgo-participants" / "data" / "donnees_demandes.csv"
    output_path = model_dir / "study_domains.csv"

    df = pd.read_csv(raw_path, usecols=["programme_etudes"])
    domains = sorted(df["programme_etudes"].dropna().astype(str).str.strip().unique())
    pd.DataFrame({"programme_etudes": domains}).to_csv(output_path, index=False)

    print(f"Found {len(domains)} unique study domains:")
    for domain in domains:
        print(domain)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
