"""The generated results page, executed in a real headless browser."""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from scripts import build_dashboard
from tests.test_build_dashboard import bind_release, dashboard_args

from naltra.utils.config import load_yaml


def renders(browser: str) -> bool:
    """Some installed builds exit at once in headless mode; use one that really renders."""
    # A browser child can still hold its profile lock; failing to remove the profile then
    # must not abort discovery of the next candidate.
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
            command = [browser, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}"]
            dom = subprocess.run(
                [*command, "--dump-dom", "data:text/html,<p id=ok>ok</p>"],
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return 'id="ok"' in dom


def find_browser() -> str | None:
    names = ("msedge", "chrome", "chromium", "chromium-browser", "google-chrome")
    roots = [os.environ.get(key) for key in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
    installs = ("Microsoft/Edge/Application/msedge.exe", "Google/Chrome/Application/chrome.exe")
    installed = [str(Path(root) / i) for root in roots if root for i in installs]
    apps = ("Google Chrome", "Microsoft Edge", "Chromium")
    installed += [f"/Applications/{app}.app/Contents/MacOS/{app}" for app in apps]
    candidates = [*filter(None, map(shutil.which, names)), *filter(os.path.exists, installed)]
    return next((browser for browser in dict.fromkeys(candidates) if renders(browser)), None)


BROWSER = find_browser()


def test_a_profile_that_cannot_be_removed_does_not_abort_browser_discovery(monkeypatch):
    class LockedProfile:
        def __enter__(self):
            return "profile"

        def __exit__(self, *exc):
            raise PermissionError("[WinError 32] lockfile is in use")

    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda **kwargs: LockedProfile())
    ok = subprocess.CompletedProcess([], 0, stdout='<p id="ok">ok</p>')
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: ok)
    assert renders("chrome") is False


pytestmark = pytest.mark.skipif(BROWSER is None, reason="needs a headless Chromium browser")
PAYLOAD = '<img src=x onerror="window.__X=1">'
SETS = ["en_test", "tr_test", "cs_chunk_test", "cs_sentence_test", "en_test_noisy", "tr_test_noisy"]
SYSTEMS = ["svm", "naive_bayes", "hard_majority", "hard_k", "hard_closed", "soft", "weighted_soft"]
SCORED = (
    "micro_f1",
    "macro_f1",
    "hmicro_f1",
    "p_at_1",
    "jaccard",
    "r_precision",
    "ece",
    "top1_ece",
    "mean_predicted_labels",
)
CATCH_ERRORS = (
    "<script>window.__errors = [];"
    " addEventListener('error', e => __errors.push(String(e.message)));</script>\n"
)
PROBE = """<script>
(async () => {
  const wait = ms => new Promise(done => setTimeout(done, ms));
  const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
  const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const choose = (id, value) => {
    $("#" + id).value = value;
    $("#" + id).dispatchEvent(new Event("change"));
  };
  const button = (legend, name) => $$(`#${legend} button`).find(b => b.textContent === name);
  const tipOf = node => {
    node.dispatchEvent(new PointerEvent("pointerenter"));
    return $("#tip").textContent;
  };
  const cells = id => $$(`#${id} tr`).map(tr => [...tr.children].map(c => c.textContent));
  const root = document.documentElement;
  const redraw = () => {
    root.dataset.theme = root.dataset.theme === "light" ? "dark" : "light";
    return wait(50);
  };
  let result;
  try { await wait(400); result = await (async () => { __PROBE__ })(); }
  catch (error) { result = {probeError: String(error && error.stack || error)}; }
  const out = document.createElement("pre");
  out.id = "probe-result";
  out.textContent = JSON.stringify({result, errors: window.__errors});
  document.body.appendChild(out);
})();
</script>
"""


def comparison(difference: float, low: float, high: float, p: float, p_holm: float) -> dict:
    stats = {"difference": difference, "ci_low": low, "ci_high": high, "p_value": p}
    return {"micro_f1": stats | {"p_resolution": 0.00019998000199980003, "p_holm": p_holm}}


FLOOR = comparison(0.0693, 0.0638, 0.0748, 0.00019998000199980003, 0.000999900009999)
SOFT_NOISY = comparison(0.00636685, 0.000443477, 0.012117442, 0.0357964, 0.1431857)
GAIN = comparison(0.02, 0.012, 0.028, 0.001, 0.004)
NULL = comparison(-0.001, -0.009, 0.007, 0.61, 1.0)


def system_metrics(index: int, system: str) -> dict:
    if system == "hard_closed":
        return {key: 0.55 if key == "hmicro_f1" else None for key in SCORED}
    return {
        key: round(0.3 + 0.04 * index + 0.01 * position, 4) for position, key in enumerate(SCORED)
    }


def system_detail(set_index: int, system: str) -> dict:
    detail = {
        "confusion": {
            "roots": ["humanities", "natural_sciences"],
            "matrix": [[0.7, 0.1], [0.05, 0.8]],
            "support": [36, 320],
        },
        "depth": {"1": round(0.8 - 0.01 * set_index, 4), "2": 0.7, "3": 0.6},
    }
    if system in ("svm", "naive_bayes", "soft", "weighted_soft") or system.startswith("extra"):
        detail["reliability"] = {
            "confidence": [0.2, 0.5, 0.9],
            "accuracy": [0.3, 0.55, 0.85],
            "count": [40, 120, 196],
        }
    return detail


def page_data(systems: list[str] = SYSTEMS) -> dict:
    cross = {system: {f"en_test->{name}": FLOOR for name in SETS[1:]} for system in ("svm", "soft")}
    cross["soft"]["en_test->en_test_noisy"] = SOFT_NOISY
    versus = {"soft": comparison(-0.15, -0.157, -0.1446, 0.0002, 0.0008), "weighted_soft": GAIN}
    latency = {"single": {"p50_ms": 1.69, "p95_ms": 2.07}, "batched": {"docs_per_second": 839.8}}
    ood = {"id_records": 4351, "ood_records": 94, "auroc": 0.9268, "fpr_at_95_tpr": 0.3149}
    return {
        "meta": {
            "generated_at": "2026-10-08 00:00 UTC",
            "commit": "abc1234",
            "labels": 473,
            "nodes": 586,
            "members": ["svm", "naive_bayes"],
            "baselines": [],
            "best_member": "svm",
            "thresholds": {"svm": 0.16, "naive_bayes": 0.22},
            "test_sets": SETS,
            "test_records": 4711,
            "pending": None,
            "notes": ["A note."],
        },
        "metrics": {
            name: {system: system_metrics(i, system) for i, system in enumerate(systems)}
            for name in SETS
        },
        "leave_one_out": {name: {} for name in SETS},
        "significance": {
            "resamples": 10000,
            "best_member": "svm",
            "reference_set": "en_test",
            "cross_condition": cross,
            "vs_best_member": {name: versus | {"hard_majority": NULL} for name in SETS},
        },
        "detail": {
            name: {
                system: system_detail(i, system) for system in systems if system != "hard_closed"
            }
            for i, name in enumerate(SETS)
        },
        "latency": {
            "samples": 200,
            "eval_set": "en_test",
            "hardware": {"cpu": "Test CPU"},
            "models": {"svm": latency | {"artifact_bytes": 2.0e8}},
        },
        "ood": {
            "heldout_root": "humanities",
            "counts": {"en_test_id": 4351},
            "results": {
                "svm": {"en": ood | {"auroc_ci_low": 0.9012, "auroc_ci_high": 0.9497}, "tr": ood}
            },
        },
    }


def execute(tmp_path: Path, page: str, probe: str) -> dict:
    """Load the page in a headless browser, run the probe after first layout, return its value."""
    page = page.replace("<head>\n", "<head>\n" + CATCH_ERRORS, 1)
    page = page.replace("</body>", PROBE.replace("__PROBE__", probe) + "</body>", 1)
    target = tmp_path / "probe.html"
    target.write_text(page, encoding="utf-8")
    command = [
        BROWSER,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        f"--user-data-dir={tmp_path / 'browser-profile'}",
        "--window-size=1280,900",
        "--virtual-time-budget=8000",
        "--dump-dom",
        target.as_uri(),
    ]
    dom = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", timeout=120
    ).stdout
    match = re.search(r'<pre id="probe-result">(.*?)</pre>', dom, re.S)
    assert match, dom[-2000:]
    output = json.loads(html.unescape(match.group(1)))
    assert output["errors"] == [], output["errors"]
    assert "probeError" not in (output["result"] or {}), output["result"]
    return output["result"]


def render(tmp_path: Path, probe: str, data: dict | None = None) -> dict:
    return execute(tmp_path, build_dashboard.render(data or page_data())[0], probe)


def test_report_strings_render_as_text_not_markup(tmp_path) -> None:
    args = dashboard_args(tmp_path)
    summary_path = Path(args[args.index("--summary") + 1])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metrics"]["test"][PAYLOAD] = dict(summary["metrics"]["test"]["svm"])
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    timing = {"single": {"p50_ms": 1.0, "p95_ms": 2.0}, "batched": {"docs_per_second": 9.0}}
    latency = {"hardware": {"cpu": PAYLOAD}, "models": {"svm": timing, "naive_bayes": timing}}
    scores = {"en": {"id_records": 9, "ood_records": 3, "auroc": 0.8, "fpr_at_95_tpr": 0.4}}
    ood = {"heldout_root": PAYLOAD, "counts": {}, "results": {"svm": scores, "nb": scores}}
    assert build_dashboard.main(bind_release(tmp_path, args, latency, ood)) == 0
    page = (tmp_path / "site/index.html").read_text(encoding="utf-8")
    result = execute(
        tmp_path,
        page,
        """
        const tips = $$("svg rect, svg circle").map(tipOf);
        await wait(300);
        return {marker: window.__X ?? null, images: $$("img").length, tips,
                tiles: $("#tiles").textContent, legend: $("#lang-legend").textContent,
                table: $("#metrics-table").textContent};
        """,
    )
    assert result["marker"] is None and result["images"] == 0
    assert PAYLOAD in result["tiles"] and "held-out " + PAYLOAD in result["tiles"]
    assert PAYLOAD in result["legend"] and PAYLOAD in result["table"]
    assert any(PAYLOAD in tip for tip in result["tips"])


def test_lede_frames_the_task_without_calling_it_topic_classification(tmp_path) -> None:
    lede = render(tmp_path, 'return $("#lede").textContent;')
    assert lede.startswith(
        "Multilingual hierarchical multi-label classification of EU research projects"
    )
    assert "EuroSciVoc science taxonomy under cross-lingual, code-switched and noisy" in lede
    assert lede.endswith("input shift, with near-OOD rejection.")
    assert "topic" not in build_dashboard.TEMPLATE.read_text(encoding="utf-8").lower()


def test_forest_colours_by_the_holm_corrected_p(tmp_path) -> None:
    result = render(
        tmp_path,
        """
        choose("sig-system", "soft");
        const fills = id => $$(`#${id} circle`).map(c => c.getAttribute("fill"));
        return {cross: fills("sig-cross"), best: fills("sig-best"),
                caption: $("#sig-sub").textContent,
                bad: cssVar("--bad"), good: cssVar("--good"), grey: cssVar("--ink-3")};
        """,
    )
    bad, good, grey = result["bad"], result["good"], result["grey"]
    assert result["cross"] == [bad, bad, bad, grey, bad]
    assert result["best"] == [bad, good, grey]
    caption = result["caption"]
    assert all(word in caption for word in ("Holm", "red", "green", "grey"))


def test_floor_note_tests_the_raw_p_against_the_resolution(tmp_path) -> None:
    result = render(
        tmp_path,
        """
        const floor = tipOf($$("#sig-cross circle")[0]);
        choose("sig-system", "soft");
        return {floor, above: tipOf($$("#sig-cross circle")[3])};
        """,
    )
    assert "≤ 0.00100 (resampling floor)" in result["floor"]
    assert "0.143" in result["above"] and "floor" not in result["above"]


def test_every_chart_is_named_and_has_a_data_table(tmp_path) -> None:
    result = render(
        tmp_path,
        """
        const ids = ["lang-chart", "sig-cross", "sig-best", "conf-chart", "depth-chart"];
        ids.push("rel-chart");
        return ids.map(id => {
          const svg = $(`#${id} svg`), details = $(`#${id}`).nextElementSibling;
          return {id, role: svg.getAttribute("role"), name: svg.getAttribute("aria-label") || "",
                  disclosure: details && details.tagName,
                  summary: details && details.querySelector("summary").textContent,
                  rows: details ? [...details.querySelectorAll("tr")]
                    .map(tr => [...tr.children].map(c => c.textContent)) : []};
        });
        """,
    )
    charts = {chart["id"]: chart for chart in result}
    for chart in result:
        assert chart["role"] == "img" and len(chart["name"]) > 20, chart
        assert chart["disclosure"] == "DETAILS" and chart["summary"] == "Show data", chart
    lang = charts["lang-chart"]["rows"]
    assert lang[0] == ["System", "English", "Turkish (MT)", "Code-switch (chunk)"] + [
        "Code-switch (sentence)",
        "English, noisy",
        "Turkish, noisy",
    ]
    assert ["SVM", *["0.3000"] * 6] in lang and len(lang) == 1 + 6
    cross = charts["sig-cross"]["rows"]
    assert cross[0] == ["Comparison", "Δ micro-F1", "95% CI low", "95% CI high"] + [
        "p (raw)",
        "p (Holm)",
    ]
    floor = "≤ 0.00100 (resampling floor)"
    assert cross[1] == ["Turkish (MT)", "-0.0693", "-0.0748", "-0.0638", "0.000200", floor]
    best = charts["sig-best"]["rows"]
    assert best[2] == ["Weighted soft vote", "0.0200", "0.0120", "0.0280", "0.00100", "0.00400"]
    depth = charts["depth-chart"]["rows"]
    assert depth[0] == ["System", "depth", "micro-F1"] and ["SVM", "1", "0.800"] in depth
    reliability = charts["rel-chart"]["rows"]
    assert reliability[0] == ["System", "top-1 confidence", "top-1 accuracy", "projects in bin"]
    assert ["SVM", "0.900", "0.850", "196"] in reliability
    confusion = charts["conf-chart"]["rows"]
    assert confusion == [
        ["Gold root", "Projects", "humanities", "natural"],
        ["humanities", "36", "0.700", "0.100"],
        ["natural", "320", "0.050", "0.800"],
    ]


def test_data_tables_follow_the_chart_controls(tmp_path) -> None:
    result = render(
        tmp_path,
        """
        choose("lang-metric", "hmicro_f1");
        choose("depth-set", "tr_test");
        choose("sig-system", "soft");
        return {lang: cells("lang-data"), depth: cells("depth-data"),
                cross: cells("sig-cross-data")};
        """,
    )
    assert ["Hierarchy vote", *["0.5500"] * 6] in result["lang"]
    assert ["SVM", "1", "0.790"] in result["depth"]
    assert result["cross"][4][-2:] == ["0.0358", "0.143"]


def test_legend_keeps_keyboard_focus_across_redraws(tmp_path) -> None:
    result = render(
        tmp_path,
        """
        button("lang-legend", "SVM").focus();
        button("lang-legend", "SVM").click();
        const after = document.activeElement;
        const clicked = {text: after.textContent, pressed: after.getAttribute("aria-pressed"),
                         inLegend: !!after.closest("#lang-legend")};
        await redraw();
        return {clicked, resized: document.activeElement.textContent,
                stillInLegend: !!document.activeElement.closest("#lang-legend")};
        """,
    )
    assert result["clicked"] == {"text": "SVM", "pressed": "false", "inLegend": True}
    assert result["resized"] == "SVM" and result["stillInLegend"]


def test_each_chart_keeps_its_own_hidden_systems(tmp_path) -> None:
    result = render(
        tmp_path,
        """
        const lines = id => $$(`#${id} polyline`).length;
        const pressed = legend => button(legend, "SVM").getAttribute("aria-pressed");
        button("lang-legend", "SVM").click();
        choose("depth-set", "tr_test");
        const afterLang = {bars: $$("#lang-chart rect").length, depthLines: lines("depth-chart"),
                           depthPressed: pressed("depth-legend")};
        button("depth-legend", "SVM").click();
        choose("lang-metric", "micro_f1");
        return {afterLang, depthLines: lines("depth-chart"), langPressed: pressed("lang-legend"),
                relPressed: pressed("rel-legend"), relLines: lines("rel-chart")};
        """,
    )
    assert result["afterLang"] == {"bars": 5 * 6, "depthLines": 6, "depthPressed": "true"}
    assert result["depthLines"] == 5 and result["langPressed"] == "false"
    assert result["relPressed"] == "true" and result["relLines"] == 4


COLOURS = """
choose("lang-metric", "hmicro_f1");
return {bars: $$("#lang-chart rect").map(r => r.getAttribute("fill")),
        swatches: $$("#lang-legend i").map(i => getComputedStyle(i).backgroundColor),
        neutral: cssVar("--ink-3"), slots: [1, 2, 3, 4, 5, 6, 7, 8].map(i => cssVar(`--s${i}`))};
"""


def test_every_system_gets_a_categorical_colour(tmp_path) -> None:
    result = render(tmp_path, COLOURS)
    per_system = [result["bars"][i] for i in range(len(SYSTEMS))]
    assert all(result["bars"]) and per_system == result["slots"][: len(SYSTEMS)]
    assert "rgba(0, 0, 0, 0)" not in result["swatches"] and len(set(result["swatches"])) == 7


def test_systems_beyond_eight_fold_to_a_neutral_colour(tmp_path) -> None:
    systems = [*SYSTEMS, "extra_1", "extra_2", "extra_3"]
    result = render(tmp_path, COLOURS, page_data(systems))
    per_system = [result["bars"][i] for i in range(len(systems))]
    assert per_system == result["slots"] + [result["neutral"]] * 2


def test_language_labels_do_not_overlap_at_phone_width(tmp_path) -> None:
    result = render(
        tmp_path,
        """
        document.body.style.width = "375px";
        await redraw();
        const boxes = $$("#lang-chart [text-anchor=middle]").map(t => t.getBoundingClientRect());
        return {width: $("#lang-chart svg").getBoundingClientRect().width,
                boxes: boxes.map(box => [box.left, box.right])};
        """,
    )
    boxes = result["boxes"]
    assert result["width"] < 320 and len(boxes) == 6
    assert all(left[1] <= right[0] for left, right in zip(boxes, boxes[1:], strict=False)), boxes


def test_ood_table_shows_the_auroc_interval_when_present(tmp_path) -> None:
    rows = render(tmp_path, 'return cells("ood-table");')
    column = rows[0].index("AUROC 95% CI")
    assert [row[column] for row in rows[1:]] == ["[0.901, 0.950]", "–"]


def test_ood_tile_reads_english_whatever_the_language_order(tmp_path) -> None:
    data = page_data()
    english = data["ood"]["results"]["svm"]["en"]
    data["ood"]["results"]["svm"] = {"tr": english | {"auroc": 0.5}, "en": english}
    result = render(
        tmp_path,
        """
        const tile = () => $$("#tiles .tile").find(t => t.textContent.includes("OOD AUROC"));
        const both = tile().textContent;
        delete DATA.ood.results.svm.en;
        staticSections();
        return {both, turkish: tile().textContent};
        """,
        data,
    )
    assert "0.927" in result["both"] and result["both"].endswith("SVM, English")
    assert "0.500" in result["turkish"] and result["turkish"].endswith("SVM, Turkish (MT)")


def test_missing_p_values_are_neither_significant_nor_at_the_floor(tmp_path) -> None:
    data = page_data()
    missing = comparison(-0.0693, -0.0748, -0.0638, None, None)
    data["significance"]["cross_condition"]["svm"]["en_test->tr_test"] = missing
    result = render(
        tmp_path,
        """
        const dot = $$("#sig-cross circle")[0];
        return {fill: dot.getAttribute("fill"), tip: tipOf(dot), grey: cssVar("--ink-3")};
        """,
        data,
    )
    assert result["fill"] == result["grey"]
    assert "floor" not in result["tip"]


def test_raw_p_at_the_floor_without_a_holm_p_shows_a_dash(tmp_path) -> None:
    data = page_data()
    missing = comparison(-0.0693, -0.0748, -0.0638, 0.00019998000199980003, None)
    data["significance"]["cross_condition"]["svm"]["en_test->tr_test"] = missing
    result = render(
        tmp_path,
        'return {tip: tipOf($$("#sig-cross circle")[0]), row: cells("sig-cross-data")[1]};',
        data,
    )
    assert result["tip"].endswith("p (Holm) –"), result["tip"]
    assert result["row"][-2:] == ["0.000200", "–"]


BENCHMARK = {
    "primary_metric": "micro_f1",
    "systems": {
        "svm": {"role": "baseline", "architecture": "Linear SVM", "model": "TF-IDF + LinearSVC"},
        "naive_bayes": {"role": "baseline", "architecture": "Naive Bayes", "model": "TF-IDF + NB"},
    }
    | {s: {"role": "new", "architecture": "Voting ensemble", "model": s} for s in SYSTEMS[2:]},
    "secondary_metrics": ["macro_f1", "hmicro_f1", "p_at_1", "top1_ece"],
    "label_distribution": {
        "tuning": {"n_records": 9422, "n_labels": 473, "mean_labels_per_record": 1.6}
        | {"rare_label_share": 0.62, "top_decile_positive_share": 0.55}
        | {"all_negative_accuracy": 0.9966},
        "en_test": {"n_records": 4711, "n_labels": 473},
    },
}
del BENCHMARK["systems"]["hard_closed"]


def benchmark_data(benchmark: dict = BENCHMARK) -> dict:
    data = page_data()
    data["benchmark"] = benchmark
    return data


def test_benchmark_ranks_systems_by_the_primary_metric_per_test_set(tmp_path) -> None:
    data = benchmark_data()
    tr = data["metrics"]["tr_test"]
    tr["svm"]["micro_f1"], tr["naive_bayes"]["micro_f1"] = 0.9, tr["hard_k"]["micro_f1"]
    result = render(
        tmp_path,
        """
        const en = cells("bench-table");
        choose("bench-set", "tr_test");
        return {en, tr: cells("bench-table")};
        """,
        data,
    )
    en, tr = result["en"], result["tr"]
    assert en[0] == ["Rank", "System", "Role", "Architecture", "Model", "Micro-F1 (primary)"] + [
        "Macro-F1",
        "Hierarchical micro-F1",
        "Precision@1",
        "Top-1 ECE",
    ]
    assert [row[:2] for row in en[1:]] == [
        ["1", "Weighted soft vote"],
        ["2", "Soft vote"],
        ["3", "k-of-M vote"],
        ["4", "Majority vote"],
        ["5", "Naive Bayes"],
        ["6", "SVM"],
        ["–", "Hierarchy vote"],
    ]
    assert en[6] == ["6", "SVM", "Baseline", "Linear SVM", "TF-IDF + LinearSVC"] + [
        "0.3000",
        "0.3100",
        "0.3200",
        "0.3300",
        "0.3700",
    ]
    assert en[2][2:5] == ["New method", "Voting ensemble", "soft"]
    assert en[7][2:7] == ["", "–", "–", "–", "–"] and en[7][7] == "0.5500"
    assert [row[:2] for row in tr[1:]] == [
        ["1", "SVM"],
        ["2", "Weighted soft vote"],
        ["3", "Soft vote"],
        ["4", "Naive Bayes"],
        ["4", "k-of-M vote"],
        ["6", "Majority vote"],
        ["–", "Hierarchy vote"],
    ]


def test_why_micro_f1_note_shows_the_label_distribution(tmp_path) -> None:
    result = render(tmp_path, 'return $("#bench-why").textContent;', benchmark_data())
    assert result.startswith("Why Micro-F1?") and "true negatives" in result
    assert "Validation: 9422 records (EN+TR) with on average 1.60 of 473 labels" in result
    assert "62.0% of labels" in result and "55.0% of all assignments" in result
    assert "scores 99.7% per-label accuracy" in result
    assert "English: 4711 projects with on average – of 473 labels" in result


def test_why_note_skips_label_distribution_entries_that_are_not_objects(tmp_path) -> None:
    extra = {"note": "text", "missing": None, "list": [1], "count": 3}
    benchmark = BENCHMARK | {"label_distribution": BENCHMARK["label_distribution"] | extra}
    result = render(tmp_path, 'return $("#bench-why").textContent;', benchmark_data(benchmark))
    assert "Validation: 9422 records (EN+TR)" in result and "English: 4711 projects" in result
    assert not any(f"{part}:" in result for part in extra), result


def test_forest_tooltips_badge_the_role_while_labels_stay_plain(tmp_path) -> None:
    result = render(
        tmp_path,
        """
        return {tips: $$("#sig-best circle").map(tipOf),
                labels: $$("#sig-best text[text-anchor=end]").map(t => t.textContent),
                name: $("#sig-best svg").getAttribute("aria-label"),
                rows: cells("sig-best-data").slice(1).map(row => row[0])};
        """,
        benchmark_data(),
    )
    plain = ["Soft vote", "Weighted soft vote", "Majority vote"]
    assert [tip.split("Δ")[0] for tip in result["tips"]] == [f"{n} New method" for n in plain]
    assert result["labels"] == plain and result["rows"] == plain
    assert "New method" not in result["name"] and ", ".join(plain) in result["name"]


def test_metrics_table_tags_non_voting_families_not_baselines(tmp_path) -> None:
    data = benchmark_data()
    data["meta"] |= {"members": ["svm"], "baselines": ["naive_bayes"]}
    result = render(
        tmp_path,
        'return {tags: $$("#metrics-table .tag").map(t => t.textContent),'
        ' legend: $$("#lang-legend button").map(b => b.textContent).slice(0, 2)};',
        data,
    )
    assert result["tags"] == ["non-voting"] + ["ensemble"] * 5
    assert result["legend"] == ["SVM Baseline", "Naive Bayes Baseline"]


def test_benchmark_table_fits_its_panel_at_1440px(tmp_path) -> None:
    systems = [*SYSTEMS, "hybrid_knn", "bilstm", "transformer", "kev"]
    data = page_data(systems)
    config = load_yaml(build_dashboard.REPO_ROOT / "configs/benchmark.yaml")
    data["benchmark"] = config | {"systems": {s: config["systems"][s] for s in systems}}
    result = render(
        tmp_path,
        """
        document.body.style.width = "1440px";
        await redraw();
        const scroller = $("#bench-table");
        return {overflow: scroller.scrollWidth - scroller.clientWidth,
                width: scroller.clientWidth};
        """,
        data,
    )
    assert result["width"] > 1000 and result["overflow"] <= 0, result


def test_role_badges_follow_system_names_in_legends_and_tooltips(tmp_path) -> None:
    result = render(
        tmp_path,
        """
        const names = id => $$(`#${id} button`).map(b => b.textContent);
        return {lang: names("lang-legend"), rel: names("rel-legend"),
                bar: tipOf($("#lang-chart rect")), dot: tipOf($("#depth-chart circle"))};
        """,
        benchmark_data(),
    )
    assert result["lang"] == [
        "SVM Baseline",
        "Naive Bayes Baseline",
        "Majority vote New method",
        "k-of-M vote New method",
        "Soft vote New method",
        "Weighted soft vote New method",
    ]
    assert result["rel"] == [
        "SVM Baseline",
        "Naive Bayes Baseline",
        "Soft vote New method",
        "Weighted soft vote New method",
    ]
    assert result["bar"].startswith("SVM Baseline · English")
    assert result["dot"].startswith("SVM Baseline")


def test_benchmark_strings_render_as_text_not_markup(tmp_path) -> None:
    entry = {"role": PAYLOAD, "architecture": PAYLOAD, "model": PAYLOAD}
    benchmark = {"primary_metric": "micro_f1", "systems": {"svm": entry}}
    result = render(
        tmp_path,
        """
        const tips = $$("#lang-chart rect").map(tipOf);
        await wait(300);
        return {marker: window.__X ?? null, images: $$("img").length, tip: tips[0],
                table: $("#bench-table").textContent, why: $("#bench-why").textContent,
                legend: $("#lang-legend").textContent};
        """,
        benchmark_data(benchmark | {"label_distribution": {PAYLOAD: {"n_records": 3}}}),
    )
    assert result["marker"] is None and result["images"] == 0
    assert result["table"].count(PAYLOAD) == 3 and PAYLOAD + ": 3 projects" in result["why"]
    assert PAYLOAD in result["legend"] and PAYLOAD in result["tip"]


LEGEND = ["SVM", "Naive Bayes", "Majority vote", "k-of-M vote", "Soft vote", "Weighted soft vote"]


def test_pages_without_a_benchmark_render_as_before(tmp_path) -> None:
    result = render(
        tmp_path,
        """
        return {display: getComputedStyle($("#p-bench")).display, tags: $$(".legend .tag").length,
                legend: $$("#lang-legend button").map(b => b.textContent),
                tip: tipOf($("#lang-chart rect"))};
        """,
    )
    assert result["display"] == "none" and result["tags"] == 0
    assert result["legend"] == LEGEND
    assert result["tip"].startswith("SVM · English")


def test_absent_sections_are_not_drawn_as_empty_panels(tmp_path) -> None:
    data = page_data()
    data["ood"] = data["latency"] = None
    result = render(
        tmp_path,
        'return ["p-ood", "p-lat", "p-sample"].map(id => getComputedStyle($("#" + id)).display);',
        data,
    )
    assert result == ["none", "none", "none"]


def sample_scores(micro: float, macro: float) -> dict:
    return {"micro_f1": micro, "macro_f1": macro}


SAMPLE = {
    "records": 120,
    "members": ["svm", "kev", "jev"],
    "headline": ["micro_f1", "macro_f1"],
    "en": {
        "svm": sample_scores(0.5557, 0.2541),
        "kev": sample_scores(0.2653, 0.1586),
        "jev": sample_scores(0.31, 0.2),
        "hard_closed": {},
    },
    "tr": {
        "svm": sample_scores(0.4893, 0.2157),
        "kev": sample_scores(0.2202, 0.1473),
        "jev": sample_scores(0.29, 0.19),
        "hard_closed": {},
    },
    "drop": {
        "svm": {"difference": 0.0664, "ci_low": 0.03, "ci_high": 0.1},
        "kev": {"difference": 0.0451, "ci_low": 0.0206, "ci_high": 0.0695},
    },
    "roles": {"svm": "baseline", "kev": "new", "jev": "new"},
}


def test_sample_table_shows_every_sample_system_in_both_languages(tmp_path) -> None:
    data = page_data()
    data["sample"] = SAMPLE
    result = render(
        tmp_path,
        """
        const scroller = $("#sample-table");
        return {display: getComputedStyle($("#p-sample")).display, rows: cells("sample-table"),
                title: $("#p-sample h2").textContent, sub: $("#sample-sub").textContent,
                overflow: scroller.scrollWidth - scroller.clientWidth};
        """,
        data,
    )
    rows = result["rows"]
    assert result["display"] != "none" and result["title"] == "120-pair English–Turkish sample"
    assert rows[0] == ["", "Micro-F1", "Micro-F1 drop (EN − TR)", "Macro-F1"]
    assert rows[1] == ["System", "EN", "TR", "Drop", "95% CI", "EN", "TR"]
    assert rows[2] == [
        "SVM Baseline",
        "0.556",
        "0.489",
        "0.066",
        "[0.030, 0.100]",
        "0.254",
        "0.216",
    ]
    assert rows[3][:5] == ["Kev-0.8B New method", "0.265", "0.220", "0.045", "[0.021, 0.070]"]
    assert rows[4] == ["Jev New method", "0.310", "0.290", "–", "–", "0.200", "0.190"]
    assert rows[5] == ["Hierarchy vote", *["–"] * 6]
    assert result["sub"].startswith("Kev-0.8B and Jev were scored only on this sample")
    assert "SVM, Kev-0.8B and Jev" in result["sub"] and "120 projects" in result["sub"]
    assert result["overflow"] <= 0


def test_benchmark_table_scrolls_inside_its_panel_at_phone_width(tmp_path) -> None:
    result = render(
        tmp_path,
        """
        document.body.style.width = "375px";
        await redraw();
        const panel = $("#p-bench"), scroller = $("#bench-table");
        return {right: panel.getBoundingClientRect().right,
                overflow: panel.scrollWidth - panel.clientWidth,
                scrolls: scroller.scrollWidth > scroller.clientWidth};
        """,
        benchmark_data(),
    )
    assert result["right"] <= 375 and result["overflow"] <= 0 and result["scrolls"]


def test_comment_opener_in_report_text_cannot_break_the_page(tmp_path) -> None:
    data = page_data()
    data["latency"]["hardware"]["cpu"] = "<!--<script>"
    page = build_dashboard.render(data)[0]
    embedded = page[page.index("const DATA = ") : page.index("const SVGNS")]
    assert "<" not in embedded
    result = execute(
        tmp_path,
        page,
        'return {bars: $$("#lang-chart rect").length, cpu: $("#lat-sub").textContent};',
    )
    assert result["bars"] > 0 and "<!--<script>" in result["cpu"]
