"""Build taxonomy/taxonomy.json and taxonomy/label_map.json from the raw CORDIS H2020 archive.

Every euroSciVoc.csv row assigns one project a direct EuroSciVoc category as a slash-delimited
code with its matching title path, so each row also spells out the category's root-to-leaf
ancestry. Categories assigned to at least --min-support distinct projects become direct labels,
and all of their ancestors are added so the active label set is a closed single-parent tree.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.4.0"
SOURCE = "European Science Vocabulary (EuroSciVoc)"
SOURCE_URI = "http://data.europa.eu/8mn/euroscivoc"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive", default="data/raw/cordis_h2020/cordis-h2020projects-csv.zip")
    result.add_argument("--output-dir", default="taxonomy")
    result.add_argument("--min-support", type=int, default=50, help="Direct projects per label")
    return result


def read_categories(archive: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Title path and directly classified project IDs of every EuroSciVoc code in the archive."""
    paths: dict[str, str] = {}
    projects: dict[str, set[str]] = defaultdict(set)
    with zipfile.ZipFile(archive) as bundle, bundle.open("euroSciVoc.csv") as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig"), delimiter=";")
        next(reader, None)
        for row in reader:
            code, path, _title, _description, project = (cell.strip() for cell in row)
            if paths.setdefault(code, path) != path:
                raise ValueError(f"EuroSciVoc code {code} has two title paths.")
            projects[code].add(project)
    return paths, projects


def lineage(code: str) -> list[str]:
    """Codes from the root down to code itself: /23/53/365 -> /23, /23/53, /23/53/365."""
    parts = code.split("/")[1:]
    return ["/" + "/".join(parts[:depth]) for depth in range(1, len(parts) + 1)]


def build(
    paths: dict[str, str], projects: dict[str, set[str]], min_support: int
) -> tuple[dict, dict]:
    """The taxonomy.json and label_map.json documents for one support threshold."""
    support = {code: len(ids) for code, ids in projects.items()}
    direct = {code for code, count in support.items() if count >= min_support}
    ancestors = {code for leaf in direct for code in lineage(leaf)[:-1]}
    path_of: dict[str, str] = {}
    for leaf in direct:
        titles = paths[leaf].split("/")
        if len(titles) != leaf.count("/"):
            raise ValueError(f"EuroSciVoc code {leaf} and its title path differ in depth.")
        for depth, code in enumerate(lineage(leaf), start=1):
            path_of[code] = "/".join(titles[:depth])
    active = sorted(path_of, key=lambda code: (code.count("/"), code))
    name = {code: path.split("/")[-1] for code, path in path_of.items()}
    label_id = {code: re.sub(r"[^a-z0-9]+", "_", name[code].lower()) for code in active}
    if len(set(label_id.values())) != len(active):
        raise ValueError("Two active EuroSciVoc categories share a label ID.")
    parents = {code: lineage(code)[-2] if code.count("/") > 1 else None for code in active}
    has_children = set(parents.values())
    labels = [
        {
            "id": label_id[code],
            "name": name[code],
            "parent": label_id.get(parents[code]),
            "description": f"EuroSciVoc category: {path_of[code]}",
            "euroscivoc_code": code,
            "euroscivoc_path": path_of[code],
            "depth": code.count("/"),
            "is_root": parents[code] is None,
            "is_leaf": code not in has_children,
            "is_direct_supported": code in direct,
            "direct_project_support": support.get(code, 0),
        }
        for code in active
    ]
    taxonomy = {
        "version": VERSION,
        "source": SOURCE,
        "source_uri": SOURCE_URI,
        "min_direct_support_threshold": min_support,
        "direct_supported_labels_count": len(direct),
        "ancestor_labels_all_count": len(ancestors),
        "ancestor_labels_already_direct_count": len(ancestors & direct),
        "ancestor_labels_added_count": len(ancestors - direct),
        "total_active_labels": len(labels),
        "root_labels_count": sum(label["is_root"] for label in labels),
        "leaf_labels_count": sum(label["is_leaf"] for label in labels),
        "labels": labels,
    }
    aliases = {
        key: label["id"]
        for label in labels
        for key in (label["euroscivoc_code"], label["euroscivoc_path"], label["name"])
    }
    return taxonomy, {"version": VERSION, "source": SOURCE, "aliases": aliases}


def unchanged(target: Path, document: dict) -> bool:
    """Whether target already holds document. Dataset manifests and trained models pin both
    files' SHA-256, and the committed label_map.json key order cannot be rebuilt (it came from
    an unseeded set), so a rebuild that changes nothing must leave the bytes alone."""
    try:
        return json.loads(target.read_text(encoding="utf-8")) == document
    except (OSError, ValueError):
        return False


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = REPO_ROOT / args.output_dir  # an absolute path replaces REPO_ROOT
    try:
        taxonomy, label_map = build(*read_categories(REPO_ROOT / args.archive), args.min_support)
        output.mkdir(parents=True, exist_ok=True)
        written = []
        for name, document in (("taxonomy.json", taxonomy), ("label_map.json", label_map)):
            if unchanged(output / name, document):
                continue
            # CRLF on every platform, like the checkout the pinned hashes were taken from.
            (output / name).write_text(
                json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\r\n"
            )
            written.append(name)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), flush=True)
        return 1
    status = {
        "status": "success",
        "labels": taxonomy["total_active_labels"],
        "direct_labels": taxonomy["direct_supported_labels_count"],
        "aliases": len(label_map["aliases"]),
        "written": written,
    }
    print(json.dumps(status), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
