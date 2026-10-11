# Experiment runbook

Run commands from the repository root in the
[configured environment](reproducibility.md#environment).

| Task | Sections |
| --- | --- |
| Recalculate metrics using installed original dumps | 0, 4 |
| Rerun inference using installed original models | 0, 3; fixed samples in 4 |
| Train fresh models | 0, 1, 2, then adapt the artifact paths in 3 |
| Build a dashboard using the archived summaries | 0, 6 |
| Regenerate documentation charts from summaries only | 6 |

Fresh outputs use separate run directories. Commands assume the original
[model folders](../models/README.md) and [prediction dumps](../results/README.md)
have been installed where needed.

## 0. Data

Restore the archived release:

```bash
python scripts/unpack_data.py
python scripts/validate_datasets.py --prebuilt-release
python scripts/prepare_data.py --dataset noisy --prebuilt-release
```

The prebuilt validator audits EN/TR records and declared split hashes, tolerating
documented translation issues. It does not perform the derived-track audit.
See [release validation](dataset.md#release-validation).

Generate the fixed EN/TR samples when recalculating sample results:

```bash
python scripts/sample_sets.py --sets en_validation tr_validation --pairs 80 --output-dir data/samples/val80
python scripts/sample_sets.py --sets en_test tr_test --pairs 120 --output-dir data/samples/test120
```

To generate a new dataset release, use a separate checkout:

```bash
python scripts/build_taxonomy.py
python scripts/prepare_data.py --dataset cordis_h2020 --download
python scripts/validate_datasets.py --languages en
python scripts/prepare_data.py --dataset translation --device cuda
python scripts/validate_datasets.py --languages en tr
python scripts/prepare_data.py --dataset noisy
python scripts/prepare_data.py --dataset code_switch
python scripts/validate_datasets.py --derived
```

This rewrites manifests and other tracked release files. It can reproduce English
and derive tracks from the resulting EN/TR data, but standard translation does
not recreate the targeted Turkish 1.1.0 repair. Give the new release a new version.
Training a freshly regenerated release omits `--prebuilt-release`.

## 1. Members

| Family | Where | Notes |
|---|---|---|
| `naive_bayes` | CPU | baseline row, does not vote |
| `svm` | CPU, ~7 min | calibrated (3-fold Platt) |
| `hybrid_knn` | GPU for the e5 encoder | TF-IDF + multilingual-e5 neighbours, reciprocal rank fusion (Cormack et al. 2009, k = 60); own run `cordis_v1.1.0_knn` |
| `bilstm` | GPU | release `cordis_v0.5.0`: head refit on the frozen encoder |
| `transformer` | GPU | `xlm-roberta-base`; release `cordis_v0.5.0`: head refit on the frozen encoder |
| `kev` | Kev server (WSL2) | zero-shot, 473 yes/no questions per document; scored on 80 validation and 120 test pairs (section 4) |
| `jev` | TypeSafe API | hosted System One model, same questions as Kev; `JEV_API_KEY`, `JEV_API_BASE_URL=https://api.typesafe.ai` in `.env` |
| `laya` | GPU in practice | zero-shot, pinned local checkpoint (`requirements-laya.txt`, `scripts/prepare_laya.py`), 473 two-option questions per document, 8 per batch: about 145 s per document on the GTX 1650, about 7 s on a teammate's RTX 4050; scored on the same samples as Kev, non-voting (0.068 EN / 0.046 TR micro-F1, section 4) |

Every locally trained member is fit on EN+TR training data, so EN -> TR measures the gap on
translated input, not zero-shot transfer; Kev, Jev and Laya are the zero-shot members.

Hardware of the reported runs:

| Step | Hardware |
|---|---|
| EN -> TR translation (historical run, about 3.9 h) | NVIDIA GeForce RTX 3060, 12 GB ([dataset guide](dataset.md#translation-review)) |
| `bilstm`, `transformer` (`cordis_v0.5.0`) | NVIDIA GeForce RTX 4050 Laptop GPU (the `hardware` field of their `naltra.json`) |
| `hybrid_knn` e5 encoding, GPU prediction dumps, Kev server | NVIDIA GeForce GTX 1650, 4 GB |
| `naive_bayes`/`svm` training, OOD, latency | Intel Core i5-9300H CPU (latency with CUDA hidden, 4 torch threads) |

The `hybrid_knn` artifact records the GPU run only as `config.device: cuda`, which its e5
encoder follows. Its `metadata.device` and `metadata.hardware` read `cpu` and `CPU` because
the shared classical trainer (`src/naltra/models/classical.py`) writes those values for every
TF-IDF family; the GPU model is not recorded in that artifact.

## 2. Train

These commands create fresh training outputs. They do not recover the original
XLM-R training lineage or guarantee identical model weights. For exact inference,
install the original artifacts instead.

```bash
python scripts/train_all.py --datasets cordis_h2020 --prebuilt-release --models naive_bayes svm --device cpu --output-dir models/retrained_v1.1.0
python scripts/train_all.py --datasets cordis_h2020 --prebuilt-release --models bilstm --device cuda --output-dir models/retrained_v0.4.0
python scripts/train_all.py --datasets cordis_h2020 --prebuilt-release --models transformer --device cuda --batch-size 16 --accumulation 1 --output-dir models/retrained_v0.4.0
python scripts/train_all.py --datasets cordis_h2020 --prebuilt-release --models bilstm transformer --device cuda --refit-head-from models/retrained_v0.4.0/cordis_h2020/en_tr_direct --max-length 512 --loss-weighting none --output-dir models/retrained_v0.5.0
python scripts/train_all.py --datasets cordis_h2020 --prebuilt-release --models hybrid_knn --device cuda --output-dir models/retrained_v1.1.0_knn
```

The `retrained_v0.4.0` checkpoints supply the encoders for the new head refit;
they are newly trained models, not replacements for the owner's archived originals.
Each run rewrites its `training_summary.json`; save the report before a later run
uses the same output directory.

On Windows, NB/SVM can exit with a native access violation (139, -1073741819,
or 0xC0000005) while joblib workers shut down, sometimes after saving the model.
Check the family's `naltra.json` and the success status in `training_summary.json`
or printed output, then verify the saved artifact by reloading it with
`predict_all.py`. If saving or reloading failed, rerun that family alone.

Training defaults select BiLSTM and Transformer. Name `--models` explicitly.
Use new directories for additional runs. Existing artifacts require explicit
`--overwrite`.

If NB/SVM fails to deserialize worker tasks under memory pressure, limit the
worker count before retrying that family:

```powershell
$env:LOKY_MAX_CPU_COUNT = "4"
```

On Linux/macOS, use `export LOKY_MAX_CPU_COUNT=4`.
A bounded offline integration check is described in [model usage](core_ml.md#setup-and-training).

## 3. Dump predictions

Using the installed original models:

```bash
python scripts/predict_all.py --artifacts models/cordis_v1.1.0/cordis_h2020/en_tr_direct --models naive_bayes svm --sets en_validation tr_validation en_test tr_test cs_chunk_test cs_sentence_test en_test_noisy=data/noisy/combined/medium/cordis_h2020/en/test.jsonl tr_test_noisy=data/noisy/combined/medium/cordis_h2020/tr/test.jsonl --device cuda --output-dir results/predictions/inference_replay
python scripts/predict_all.py --artifacts models/cordis_v1.1.0_knn/cordis_h2020/en_tr_direct --models hybrid_knn --sets en_validation tr_validation en_test tr_test cs_chunk_test cs_sentence_test en_test_noisy=data/noisy/combined/medium/cordis_h2020/en/test.jsonl tr_test_noisy=data/noisy/combined/medium/cordis_h2020/tr/test.jsonl --device cuda --output-dir results/predictions/inference_replay
python scripts/predict_all.py --artifacts models/cordis_v0.5.0/cordis_h2020/en_tr_direct --models bilstm transformer --sets en_validation tr_validation en_test tr_test cs_chunk_test cs_sentence_test en_test_noisy=data/noisy/combined/medium/cordis_h2020/en/test.jsonl tr_test_noisy=data/noisy/combined/medium/cordis_h2020/tr/test.jsonl --device cuda --output-dir results/predictions/inference_replay
```

These commands write new dumps to `results/predictions/inference_replay/`.
Compare them with the archived originals before using them for a new evaluation.
For fresh training outputs, adapt `--artifacts` to the paths from section 2.

Every NPZ matrix has a JSON sidecar identifying inputs, label order, record count,
and artifact fingerprints. Prediction generation checks batch size/order, model
identity, label completeness, and finite probability values before writing.

Use `--limit 50` for a pilot. Pilot dumps are incomplete and cannot reproduce
full-set metrics. Kev/Jev/Laya are evaluated on samples, rather than these full sets.

## 4. Benchmark

### Full results from archived predictions

```bash
python scripts/evaluate_ensemble.py --predictions results/predictions/cordis_v1.1.0 --members svm hybrid_knn bilstm transformer --baselines naive_bayes --tune-sets en_validation tr_validation --test-sets en_test tr_test cs_chunk_test cs_sentence_test en_test_noisy=data/noisy/combined/medium/cordis_h2020/en/test.jsonl tr_test_noisy=data/noisy/combined/medium/cordis_h2020/tr/test.jsonl --output-dir results/metrics/replay_full
python scripts/make_figures.py --summary results/metrics/replay_full/summary.json --predictions results/predictions/cordis_v1.1.0 --test-sets en_test tr_test cs_chunk_test --output-dir results/plots/replay
```

New full-set summaries go to `results/metrics/replay_full/`. Compare them against
`results/metrics/cordis_v1.1.0_ensemble/`. Plots also use a separate output location.

The benchmark partitions validation by pair identity, tunes member thresholds on
val-A and ensemble settings on val-B, then scores test. Methods and uncertainty
are explained in [evaluation](evaluation.md).

Optional translation-exception analysis additionally needs the original SVM
`naltra.json` for its prebuilt-audit provenance:

```bash
python scripts/translation_exceptions.py --summary results/metrics/replay_full/summary.json --predictions results/predictions/cordis_v1.1.0 --sets en_test tr_test cs_chunk_test cs_sentence_test en_test_noisy=data/noisy/combined/medium/cordis_h2020/en/test.jsonl tr_test_noisy=data/noisy/combined/medium/cordis_h2020/tr/test.jsonl --artifact models/cordis_v1.1.0/cordis_h2020/en_tr_direct/svm/naltra.json --output results/metrics/replay_translation_exceptions.json
```

### Kev, Jev and Laya on fixed samples

Using the archived sample predictions:

```bash
python scripts/evaluate_ensemble.py --predictions results/predictions/cordis_v1.1.0 --members svm hybrid_knn bilstm transformer kev jev --baselines naive_bayes laya --tune-sets en_val_kev=data/samples/val80/en_validation.jsonl tr_val_kev=data/samples/val80/tr_validation.jsonl --test-sets en_test_kev=data/samples/test120/en_test.jsonl tr_test_kev=data/samples/test120/tr_test.jsonl --output-dir results/metrics/replay_sample
```

Compare `results/metrics/replay_sample/` with
`results/metrics/cordis_v1.1.0_sample/`. This recalculation needs no API access.

To generate new sample predictions, start the Kev server, configure Jev credentials,
then run:

```bash
python scripts/predict_all.py --artifacts models/cordis_v1.1.0/cordis_h2020/en_tr_direct --models naive_bayes svm kev jev --sets en_val_kev=data/samples/val80/en_validation.jsonl tr_val_kev=data/samples/val80/tr_validation.jsonl en_test_kev=data/samples/test120/en_test.jsonl tr_test_kev=data/samples/test120/tr_test.jsonl --device cuda --output-dir results/predictions/inference_replay
python scripts/predict_all.py --artifacts models/cordis_v1.1.0_knn/cordis_h2020/en_tr_direct --models hybrid_knn --sets en_val_kev=data/samples/val80/en_validation.jsonl tr_val_kev=data/samples/val80/tr_validation.jsonl en_test_kev=data/samples/test120/en_test.jsonl tr_test_kev=data/samples/test120/tr_test.jsonl --device cuda --output-dir results/predictions/inference_replay
python scripts/predict_all.py --artifacts models/cordis_v0.5.0/cordis_h2020/en_tr_direct --models bilstm transformer --sets en_val_kev=data/samples/val80/en_validation.jsonl tr_val_kev=data/samples/val80/tr_validation.jsonl en_test_kev=data/samples/test120/en_test.jsonl tr_test_kev=data/samples/test120/tr_test.jsonl --device cuda --output-dir results/predictions/inference_replay
```

Sample generation in section 0 selects 80 validation and 120 test pairs by
smallest pair-ID hash. EN/TR contain the same projects, and every model is evaluated
on those same files. See [sample coverage](evaluation.md#metrics).

Optional Laya preparation and sample inference:

```bash
python scripts/prepare_laya.py --output-dir models/cordis_laya_v1.1.0/cordis_h2020/en_tr_direct
python scripts/predict_all.py --artifacts models/cordis_laya_v1.1.0/cordis_h2020/en_tr_direct --models laya --sets en_val_kev=data/samples/val80/en_validation.jsonl tr_val_kev=data/samples/val80/tr_validation.jsonl en_test_kev=data/samples/test120/en_test.jsonl tr_test_kev=data/samples/test120/tr_test.jsonl --device cuda --output-dir results/predictions/inference_replay
```

The recorded Laya sample scores are 0.068 English / 0.046 Turkish micro-F1
(threshold 0.98, tuned on val-A). The sample evaluator includes its archived dumps
with `--baselines naive_bayes laya`.
Retain the same voting members. New inference dumps use `inference_replay/`;
change the evaluator's `--predictions` when evaluating that run.

### Efficiency and OOD

Remeasure latency using installed original artifacts on an idle machine:

```bash
python scripts/run_benchmark.py --artifacts models/cordis_v1.1.0/cordis_h2020/en_tr_direct models/cordis_v1.1.0_knn/cordis_h2020/en_tr_direct models/cordis_v0.5.0/cordis_h2020/en_tr_direct --models naive_bayes svm hybrid_knn bilstm transformer --samples 200 --batch-size 64 --device cpu --output-dir results/benchmarks/replay
```

The original latency settings used CPU inference with CUDA hidden and four torch
threads. Different hardware/thread settings produce different timings.

For a new humanities-held-out training/evaluation run:

```bash
python scripts/run_ood.py --models naive_bayes svm --languages en tr --output-dir results/ood/replay
```

This retrains separate models. The original OOD model folders are described in
[model artifacts](../models/README.md). `run_ood.py` replaces a recognized prior
run in its output directory, rejects unrelated existing directories unless
`--overwrite` is supplied, and rejects overlap with data or the repository root.

## 5. Kev server (WSL2, Ubuntu)

```bash
python3 -m pip install --user uv   # where pip refuses system installs (Ubuntu 23.04+): pipx install uv
git clone https://github.com/jaredpalmer/kev.git && cd kev
git rev-parse HEAD                 # record it: the reported run's server commit was not recorded
uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b@bf75a6a8848ea6960ff2ed108d9ed44c2941174f --port 8009
```

The reported Kev is `jaredpalmer/kev-0.8b` at revision
`bf75a6a8848ea6960ff2ed108d9ed44c2941174f`, on `Qwen/Qwen3.5-0.8B-Base` @
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. Both are Hugging Face Hub revisions of the
weights. The reported run named the tag `v1.0` (`kev-0.8b@v1.0`), which resolved to that
revision when it ran; the command above names the revision itself, so a moved tag cannot
change the weights. Check the served revision before rescoring. The commit of the
`jaredpalmer/kev` server code that served the run was not recorded. Set
`KEV_API_KEY=local` (the server ignores it) and `KEV_API_BASE_URL=http://127.0.0.1:8009` in
`.env`. Starting the server with `KEV_DTYPE=fp16` made no difference.

Kev-0.8B ran on a GTX 1650 (4 GB). Its adapter is English-only, so its Turkish results are
zero-shot cross-lingual transfer; the questions stay English. Kev (like Jev) is scored only on
the EN and TR sample sets of section 4, not on the code-switch or noisy sets.

## 6. Interactive results page

The [results guide](results.md) embeds eight static charts. Regenerate their PNG
and SVG versions directly from the committed summaries, without prediction dumps
or model artifacts:

```bash
python scripts/plot_results.py
```

Outputs go to `docs/figures/cordis_v1.1.0/`, including a source-checksum manifest.
This redraws archived measurements; it does not rerun the experiments.

Use the archived metric, sample, latency, and OOD summaries plus the matching
prediction dumps:

```bash
python scripts/build_dashboard.py --summary results/metrics/cordis_v1.1.0_ensemble/summary.json --predictions results/predictions/cordis_v1.1.0 --test-sets en_test tr_test cs_chunk_test cs_sentence_test en_test_noisy=data/noisy/combined/medium/cordis_h2020/en/test.jsonl tr_test_noisy=data/noisy/combined/medium/cordis_h2020/tr/test.jsonl --latency results/benchmarks/cordis_v1.1.0/benchmark_summary.json --ood results/ood/cordis_v1.1.0/summary.json --sample-summary results/metrics/cordis_v1.1.0_sample/summary.json --output results/replay_dashboard/index.html
```

Open `results/replay_dashboard/index.html` in a browser. To visualize new runs,
replace summary/prediction inputs with the corresponding replay outputs. The builder
rejects dumps not scored by the summary or a different taxonomy.

The committed page is `docs/results/index.html`. It is self-contained and can be
opened directly. To serve it locally:

```bash
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000/results/`. Stop the server with Ctrl+C.
A documentation-folder static host can serve the same page without an application backend.

## Script reference

All scripts run from the repository root. Use `python scripts/<name>.py --help`
for options.

| Script | Inputs -> outputs |
| --- | --- |
| `unpack_data.py` | `data/processed/**/*.jsonl.gz` + manifests -> verified `.jsonl` splits |
| `build_taxonomy.py` | Raw archive -> `taxonomy/taxonomy.json` and `taxonomy/label_map.json` |
| `prepare_data.py` | Raw archive -> EN splits; `--dataset translation`, `code_switch` or `all` for derived tracks; `--dataset noisy` -> `data/noisy/combined/medium/cordis_h2020/{en,tr}/test.jsonl`; `--prebuilt-release` derives from the shipped 1.1.0 release |
| `validate_datasets.py` | Prepared release -> full audit pass or error; `--code-switch` adds both code-switch tracks, `--derived` noise and code-switch; `--prebuilt-release` audits the shipped 1.1.0 release record by record, which the full audit refuses ([why](dataset.md#release-validation)) |
| `train_all.py` | Splits + `configs/` -> `models/<run>/cordis_h2020/<track>/<family>/`; rewrites `<run>/training_summary.json` on every run |
| `prepare_laya.py` | Pinned checkpoint in `configs/models/laya.yaml` -> `<output-dir>/laya/` artifact (checkpoint files with their SHA-256s, the 473 topic questions); no training |
| `sample_sets.py` | Evaluation sets -> the `--pairs` projects with the smallest pair-id SHA-256 in each, `data/samples/<dir>/<set>.jsonl`, for Kev, Jev and Laya |
| `predict_all.py` | Artifacts + evaluation sets -> score matrices `results/predictions/<run>/<family>/<set>.npz` with `.json` sidecars |
| `classify.py` | Artifacts + `--text` or a JSONL `--input` -> one JSON line per text with labels, taxonomy-closed labels, language and OOD decision; `--benchmark-settings summary.json` loads frozen member thresholds and hard, tuned k-of-M, soft or weighted-soft votes after checking artifact/taxonomy hashes. Without it, artifact thresholds and the default 0.5 ensemble threshold apply. Ensemble OOD remains disabled. |
| `evaluate_ensemble.py` | Score matrices -> tuned thresholds, member and ensemble metrics, bootstrap tests in `results/metrics/<run>/summary.{json,md}` |
| `evaluate_all.py` | Direct BiLSTM/XLM-R/Laya evaluator with `--resume`; it runs the full validator, so it needs a release rebuilt from the raw archive, and no reported number comes from it |
| `translation_exceptions.py` | Summary + dumps + a `--prebuilt-release` artifact -> the projects the prebuilt audit flagged (0 on 1.1.0) and scores with and without them, `results/metrics/translation_exceptions.json` |
| `make_figures.py` | Summary + score matrices -> paper figures `results/plots/<run>/*.png` |
| `plot_results.py` | Full, sample, latency and OOD summaries -> eight PNG/SVG documentation charts plus source-checksum manifest in `docs/figures/cordis_v1.1.0/`; no model files or dumps required |
| `run_benchmark.py` | Artifacts -> latency, throughput and size in `results/benchmarks/<run>/benchmark_summary.*` |
| `run_ood.py` | Splits -> retrains NB/SVM without one root (default humanities), near-OOD AUROC and FPR@95%TPR in `<output-dir>/summary.json` (reported: `results/ood/cordis_v1.1.0`); refuses an existing `--output-dir` unless it holds an earlier `run_ood.py` output or `--overwrite` is given, and always refuses one that overlaps `--data-dir` or the repository root |
| `build_dashboard.py` | Summaries + dumps + latency + OOD -> a standalone results page |

## Related guides

[Results and limitations](results.md) · [Reproducibility](reproducibility.md) ·
[Dataset](dataset.md) · [Model usage](core_ml.md)
