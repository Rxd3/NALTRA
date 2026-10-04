import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

DATASET_PATH = Path("data/raw/mn_ds/MN-DS-news-classification.csv")
TAXONOMY_PATH = Path("taxonomy/taxonomy.json")
LABEL_MAP_PATH = Path("taxonomy/label_map.json")


LEVEL_1_MAP = {
    "arts, culture, entertainment and media": "arts_culture_entertainment_media",
    "conflict, war and peace": "conflict_war_peace",
    "crime, law and justice": "crime_law_justice",
    "disaster, accident and emergency incident": "disaster_accident_emergency",
    "economy, business and finance": "economy_business_finance",
    "education": "education",
    "environment": "environment",
    "health": "health",
    "human interest": "human_interest",
    "labour": "labour",
    "lifestyle and leisure": "lifestyle_leisure",
    "politics": "politics",
    "religion and belief": "religion_belief",
    "science and technology": "science_technology",
    "society": "society",
    "sport": "sport",
    "weather": "weather",
}


def make_id(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def add_alias(
    aliases: dict[str, str],
    source: str,
    target: str,
) -> None:
    existing = aliases.get(source)

    if existing is not None and existing != target:
        raise ValueError(f"Alias conflict for '{source}': " f"{existing} != {target}")

    aliases[source] = target


def main() -> None:
    print("Loading MN-DS...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"MN-DS file not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    pairs = df[["category_level_1", "category_level_2"]].drop_duplicates()

    # Make sure every level-2 label has exactly one parent.
    parent_counts = pairs.groupby("category_level_2")["category_level_1"].nunique()

    ambiguous = parent_counts[parent_counts > 1]

    if not ambiguous.empty:
        raise ValueError("Some MN-DS level-2 labels have multiple parents:\n" f"{ambiguous}")

    unknown_level_1 = set(pairs["category_level_1"]) - set(LEVEL_1_MAP)

    if unknown_level_1:
        raise ValueError(f"Unknown level-1 categories: {unknown_level_1}")

    with TAXONOMY_PATH.open(encoding="utf-8") as file:
        taxonomy = json.load(file)

    with LABEL_MAP_PATH.open(encoding="utf-8") as file:
        label_map = json.load(file)

    labels = taxonomy["labels"]
    aliases = label_map["aliases"]

    existing_labels = {label["id"]: label for label in labels}

    # Verify all canonical parent labels exist.
    for parent_id in LEVEL_1_MAP.values():
        if parent_id not in existing_labels:
            raise ValueError(f"Missing canonical parent: {parent_id}")

    # Add MN-DS level-1 aliases.
    for source_label, canonical_id in LEVEL_1_MAP.items():
        add_alias(
            aliases,
            source_label,
            canonical_id,
        )

    added = 0
    reused = 0

    # Add all 109 MN-DS level-2 labels.
    for _, row in pairs.sort_values(["category_level_1", "category_level_2"]).iterrows():
        source_parent = row["category_level_1"]
        source_label = row["category_level_2"]

        parent_id = LEVEL_1_MAP[source_parent]
        label_id = make_id(source_label)

        if label_id in existing_labels:
            existing = existing_labels[label_id]

            if existing.get("parent") != parent_id:
                raise ValueError(
                    f"Label ID conflict: {label_id} "
                    f"has parent {existing.get('parent')} "
                    f"but MN-DS expects {parent_id}"
                )

            reused += 1
        else:
            new_label = {
                "id": label_id,
                "name": source_label.title(),
                "parent": parent_id,
                "description": ("MN-DS fine-grained topic under " f"{source_parent}."),
            }

            labels.append(new_label)
            existing_labels[label_id] = new_label
            added += 1

        add_alias(
            aliases,
            source_label,
            label_id,
        )

    with TAXONOMY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            taxonomy,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    with LABEL_MAP_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            label_map,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    print("\n=== MN-DS TAXONOMY UPDATE ===")
    print(f"MN-DS level-2 labels: {len(parent_counts)}")
    print(f"New canonical labels added: {added}")
    print(f"Existing labels reused: {reused}")
    print(f"Total canonical labels: {len(labels)}")


if __name__ == "__main__":
    main()
