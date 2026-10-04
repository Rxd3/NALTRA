from pathlib import Path

import pandas as pd

DATASET_PATH = Path("data/raw/mn_ds/MN-DS-news-classification.csv")


def main() -> None:
    print("Loading MN-DS...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    print("\n=== DATASET SIZE ===")
    print(f"Rows: {len(df)}")
    print(f"Unique articles: {df['id'].nunique()}")
    print(f"Columns: {len(df.columns)}")

    print("\n=== MISSING DATA ===")
    print(f"Missing titles: {df['title'].isna().sum()}")
    print(f"Missing content: {df['content'].isna().sum()}")
    print("Missing level-1 labels: " f"{df['category_level_1'].isna().sum()}")
    print("Missing level-2 labels: " f"{df['category_level_2'].isna().sum()}")

    print("\n=== TAXONOMY SIZE ===")
    print("Level-1 categories: " f"{df['category_level_1'].nunique()}")
    print("Level-2 categories: " f"{df['category_level_2'].nunique()}")

    print("\n=== LEVEL-1 DISTRIBUTION ===")

    level_1_counts = df["category_level_1"].value_counts()

    for label, count in level_1_counts.items():
        print(f"{label}: {count}")

    print("\n=== LEVEL-2 DISTRIBUTION ===")

    level_2_counts = df["category_level_2"].value_counts()

    print(f"Largest category: {level_2_counts.max()}")
    print(f"Smallest category: {level_2_counts.min()}")
    print(f"Median category size: {level_2_counts.median():.1f}")

    print("Categories with fewer than 20 examples: " f"{(level_2_counts < 20).sum()}")

    print("Categories with fewer than 50 examples: " f"{(level_2_counts < 50).sum()}")

    print("\n=== MULTI-LABEL ANALYSIS ===")

    # Remove any exact duplicate label assignments before counting.
    annotations = df[["id", "category_level_1", "category_level_2"]].drop_duplicates()

    annotations_per_article = annotations.groupby("id").size()

    single_label_articles = (annotations_per_article == 1).sum()

    multi_label_articles = (annotations_per_article > 1).sum()

    total_articles = len(annotations_per_article)

    multi_label_percentage = multi_label_articles / total_articles * 100

    print(f"Articles with one annotation: {single_label_articles}")
    print(
        "Articles with multiple annotations: "
        f"{multi_label_articles} "
        f"({multi_label_percentage:.2f}%)"
    )
    print("Maximum annotations for one article: " f"{annotations_per_article.max()}")

    print("\n=== MULTI-LABEL TYPE ===")

    level_1_per_article = annotations.groupby("id")["category_level_1"].nunique()

    level_2_per_article = annotations.groupby("id")["category_level_2"].nunique()

    multiple_level_1 = (level_1_per_article > 1).sum()
    multiple_level_2 = (level_2_per_article > 1).sum()

    print("Articles spanning multiple level-1 categories: " f"{multiple_level_1}")

    print("Articles with multiple level-2 categories: " f"{multiple_level_2}")


if __name__ == "__main__":
    main()
