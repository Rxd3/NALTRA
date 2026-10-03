from collections import Counter

from datasets import load_dataset


DATASET_NAME = "Davlan/sib200"
LANGUAGES = {
    "English": "eng_Latn",
    "Turkish": "tur_Latn",
}


def main() -> None:
    datasets = {}

    print("Loading SIB-200...")

    for language, config in LANGUAGES.items():
        datasets[language] = load_dataset(DATASET_NAME, config)

    print("\n=== SPLIT SIZES ===")

    for language, dataset in datasets.items():
        print(f"\n{language}")

        total = 0

        for split_name, split in dataset.items():
            print(f"{split_name}: {len(split)}")
            total += len(split)

        print(f"Total: {total}")

    print("\n=== CATEGORY DISTRIBUTION ===")

    for language, dataset in datasets.items():
        print(f"\n{language}")

        all_categories = Counter(
            category
            for split in dataset.values()
            for category in split["category"]
        )

        for category, count in all_categories.most_common():
            print(f"{category}: {count}")

    english = datasets["English"]
    turkish = datasets["Turkish"]

    print("\n=== ENGLISH / TURKISH ALIGNMENT ===")

    for split_name in english:
        id_match = (
            english[split_name]["index_id"]
            == turkish[split_name]["index_id"]
        )

        category_match = (
            english[split_name]["category"]
            == turkish[split_name]["category"]
        )

        print(
            f"{split_name}: "
            f"IDs match={id_match}, "
            f"categories match={category_match}"
        )

    print("\n=== UNIQUE IDS ===")

    for split_name, split in turkish.items():
        unique_ids = len(set(split["index_id"]))

        print(
            f"{split_name}: "
            f"{len(split)} rows, "
            f"{unique_ids} unique IDs"
        )

    print("\n=== SPLIT OVERLAP ===")

    train_ids = set(turkish["train"]["index_id"])
    validation_ids = set(turkish["validation"]["index_id"])
    test_ids = set(turkish["test"]["index_id"])

    print(
        "train / validation:",
        len(train_ids & validation_ids),
    )

    print(
        "train / test:",
        len(train_ids & test_ids),
    )

    print(
        "validation / test:",
        len(validation_ids & test_ids),
    )


if __name__ == "__main__":
    main()