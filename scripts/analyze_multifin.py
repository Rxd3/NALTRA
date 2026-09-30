from collections import Counter

from datasets import load_dataset


DATASET_NAME = "awinml/MultiFin"
CONFIG_NAME = "all_languages_lowlevel"
TARGET_LANGUAGES = {"English", "Turkish"}


def main() -> None:
    print("Loading MultiFin...")
    dataset = load_dataset(DATASET_NAME, CONFIG_NAME)

    print("\n=== SPLIT SIZES ===")
    total = 0

    for split_name, split in dataset.items():
        print(f"{split_name}: {len(split)}")
        total += len(split)

    print(f"Total: {total}")

    print("\n=== LANGUAGE DISTRIBUTION ===")

    language_counts = Counter(
        row["lang"]
        for split in dataset.values()
        for row in split
    )

    for language, count in language_counts.most_common():
        print(f"{language}: {count}")

    filtered_rows = [
        row
        for split in dataset.values()
        for row in split
        if row["lang"] in TARGET_LANGUAGES
    ]

    print("\n=== ENGLISH + TURKISH DATA ===")
    print(f"Total EN/TR samples: {len(filtered_rows)}")

    target_language_counts = Counter(
        row["lang"] for row in filtered_rows
    )

    for language, count in target_language_counts.items():
        percentage = count / len(filtered_rows) * 100
        print(f"{language}: {count} ({percentage:.2f}%)")

    print("\n=== LABEL DISTRIBUTION ===")

    label_counts = Counter(
        label
        for row in filtered_rows
        for label in row["labels"]
    )

    print(f"Unique labels: {len(label_counts)}")

    for label, count in label_counts.most_common():
        print(f"{label}: {count}")

    print("\n=== MULTI-LABEL DISTRIBUTION ===")

    labels_per_example = Counter(
        len(row["labels"]) for row in filtered_rows
    )

    for number_of_labels, count in sorted(labels_per_example.items()):
        percentage = count / len(filtered_rows) * 100
        print(
            f"{number_of_labels} label(s): "
            f"{count} ({percentage:.2f}%)"
        )

    multi_label_count = sum(
        count
        for number_of_labels, count in labels_per_example.items()
        if number_of_labels > 1
    )

    multi_label_percentage = (
        multi_label_count / len(filtered_rows) * 100
    )

    print(
        f"Multi-label examples: "
        f"{multi_label_count} ({multi_label_percentage:.2f}%)"
    )

    print("\n=== ENGLISH VS TURKISH LABEL COUNTS ===")

    english_counts = Counter(
        label
        for row in filtered_rows
        if row["lang"] == "English"
        for label in row["labels"]
    )

    turkish_counts = Counter(
        label
        for row in filtered_rows
        if row["lang"] == "Turkish"
        for label in row["labels"]
    )

    all_labels = sorted(
        set(english_counts) | set(turkish_counts)
    )

    for label in all_labels:
        print(
            f"{label:40} "
            f"EN: {english_counts[label]:4} "
            f"TR: {turkish_counts[label]:4}"
        )


if __name__ == "__main__":
    main()