import pandas as pd

def find_semantic_best_seeds(csv_file, min_other_success=1):
    # Load CSV and handle extra spaces
    df = pd.read_csv(csv_file, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    # Filter out TIMEOUTs
    success_df = df[df["Result"].str.strip() != "TIMEOUT"].copy()

    # Ensure Frames is numeric
    success_df["Frames"] = pd.to_numeric(success_df["Frames"], errors="coerce")

    result_seeds = []

    for seed, group in success_df.groupby("Seed"):
        # Count other successful algorithms (excluding SemanticScorer)
        other_success = group[group["Algorithm"].str.strip() != "SemanticScorer"]["Algorithm"].nunique()

        # Skip if not enough other successful algorithms
        if other_success < min_other_success:
            continue

        # Find minimum frames
        min_frames = group["Frames"].min()

        # Get algorithms that achieved this minimum
        fastest = group[group["Frames"] == min_frames]

        # Only consider seeds where SemanticScorer is the **sole fastest**
        if len(fastest) == 1 and fastest.iloc[0]["Algorithm"].strip() == "SemanticScorer":
            result_seeds.append(seed)

    return result_seeds


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python script.py input.csv [min_other_success]")
        exit(1)

    csv_file = sys.argv[1]
    min_other = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    seeds = find_semantic_best_seeds(csv_file, min_other_success=min_other)

    print(f"Seeds where SemanticScorer is strictly fastest and at least {min_other} other algorithms succeeded:")
    for s in seeds:
        print(s)