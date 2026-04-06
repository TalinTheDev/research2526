import pandas as pd

def find_semantic_best_seeds(csv_file, min_other_success=1):
    # Load CSV
    df = pd.read_csv(csv_file, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    # Normalize strings
    df["Result"] = df["Result"].str.strip()
    df["Algorithm"] = df["Algorithm"].str.strip()

    # Keep only successful runs
    success_df = df[df["Result"] != "TIMEOUT"].copy()

    # Ensure Frames is numeric
    success_df["Frames"] = pd.to_numeric(success_df["Frames"], errors="coerce")

    result_seeds = []

    for seed, group in success_df.groupby("Seed"):
        # Check SemanticScorer exists and succeeded
        if "SemanticScorer" not in group["Algorithm"].values:
            continue

        # Split SemanticScorer vs others
        semantic_row = group[group["Algorithm"] == "SemanticScorer"].iloc[0]
        other_group = group[group["Algorithm"] != "SemanticScorer"]

        # Count OTHER successful algorithms
        if len(other_group) < min_other_success:
            continue

        # SemanticScorer must be strictly faster than ALL others
        if all(semantic_row["Frames"] < other_group["Frames"]):
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

    print(f"{len(seeds)} Seeds where SemanticScorer is strictly best and beats at least {min_other} other successful algorithms:")
    for s in seeds:
        print(s)