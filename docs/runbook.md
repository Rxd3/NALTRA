# Experiment runbook

How the paper numbers are produced, end to end. Every model scores every evaluation set
once; thresholds, ensembles and metrics are then computed offline from those dumps.

```text
unpack_data.py ─► data/processed/cordis_h2020/**/*.jsonl              (verified splits)
train_all.py ──► models/<run>/cordis_h2020/en_tr_direct/<family>/      (artifacts)
predict_all.py ─► results/predictions/<run>/<family>/<set>.npz + .json  (score matrices)
evaluate_ensemble.py ► results/metrics/<run>/summary.{json,md}          (tables + CIs)
make_figures.py ► results/plots/<run>/*.png                            (figures)
build_dashboard.py ► docs/results/index.html                           (interactive page)
run_benchmark.py ► results/benchmarks/<run>/benchmark_summary.*        (latency, size)
```

## 0. Data

```bash
python scripts/unpack_data.py      # or: make data
python scripts/validate_datasets.py --prebuilt-release
```

This restores the committed release under `data/processed/cordis_h2020/` (`en/`, `tr/`,
`code_switch/`), release 1.1.0, and audits it record by record. The plain validator refuses
the shipped release ([why](dataset.md#shipped-turkish-release-110)), so training and noisy
generation also take `--prebuilt-release`, which records tolerated translation issues in each
artifact's provenance.

EN can be rebuilt from the raw archive with the committed scripts, and so can the derived
tracks (noisy copies, code-switch) from the EN/TR release they are built on. TR 1.1.0
cannot: it is a frozen imported release. Its targeted repair of 598 records in 128-token
source chunks (recorded in the TR manifest's `generation_parameters`) was run by a
teammate, and the repair script is not in the repository, so a standard translation run
builds a different TR release that needs a new release version.

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
| `laya` | GPU in practice | zero-shot, pinned local checkpoint (`requirements-laya.txt`, `scripts/prepare_laya.py`), 473 two-option questions per document, 8 per batch: about 145 s per document on the GTX 1650, about 7 s on a teammate's RTX 4050; scored on the same samples as Kev, results pending (section 4) |

Every locally trained member is fit on EN+TR training data, so EN -> TR measures the gap on
translated input, not zero-shot transfer; Kev, Jev and Laya are the zero-shot members.

Hardware of the reported runs:

| Step | Hardware |
|---|---|
| EN -> TR translation (README step 8, about 3.9 h) | NVIDIA GeForce RTX 3060, 12 GB ([dataset card](dataset.md#gpu-compute-environment)) |
| `bilstm`, `transformer` (`cordis_v0.5.0`) | NVIDIA GeForce RTX 4050 Laptop GPU (the `hardware` field of their `naltra.json`) |
| `hybrid_knn` e5 encoding, GPU prediction dumps, Kev server | NVIDIA GeForce GTX 1650, 4 GB |
| `naive_bayes`/`svm` training, OOD, latency | Intel Core i5-9300H CPU (latency with CUDA hidden, 4 torch threads) |

The `hybrid_knn` artifact records the GPU run only as `config.device: cuda`, which its e5
encoder follows. Its `metadata.device` and `metadata.hardware` read `cpu` and `CPU` because
the shared classical trainer (`src/naltra/models/classical.py`) writes those values for every
TF-IDF family; the GPU model is not recorded in that artifact.

## 2. Train

```bash
python scripts/train_all.py --datasets cordis_h2020 --prebuilt-release --models naive_bayes svm --device cpu --output-dir models/cordis_v1.1.0
python scripts/train_all.py --datasets cordis_h2020 --prebuilt-release --models bilstm --device cuda --output-dir models/cordis_v0.4.0
python scripts/train_all.py --datasets cordis_h2020 --prebuilt-release --models transformer --device cuda --batch-size 16 --accumulation 1 --output-dir models/cordis_v0.4.0
python scripts/train_all.py --datasets cordis_h2020 --prebuilt-release --models bilstm transformer --device cuda --refit-head-from models/cordis_v0.4.0/cordis_h2020/en_tr_direct --max-length 512 --loss-weighting none --output-dir models/cordis_v0.5.0
python scripts/train_all.py --datasets cordis_h2020 --prebuilt-release --models hybrid_knn --device cuda --output-dir models/cordis_v1.1.0_knn
```

Each command rewrites `<output-dir>/training_summary.json`; check it says `"success"` before
the next run overwrites it. The `--refit-head-from` command builds the `cordis_v0.5.0` neural
release; the teammate's run used this same TR 1.1.0 release. `hybrid_knn` gets its own run
directory; its dense e5 encoder needs a GPU in practice.

If `naive_bayes` or `svm` training fails with joblib's "A task has failed to un-serialize"
(seen under memory pressure, since both train with `n_jobs: -1`), set
`LOKY_MAX_CPU_COUNT=4` and re-run that family.

## 3. Dump predictions

Noisy test copies (`combined`: character swap, duplication or deletion plus casing and
punctuation noise; medium, seed 42):

```bash
python scripts/prepare_data.py --dataset noisy --prebuilt-release
python scripts/predict_all.py --artifacts models/cordis_v1.1.0/cordis_h2020/en_tr_direct --models naive_bayes svm --sets en_validation tr_validation en_test tr_test cs_chunk_test cs_sentence_test en_test_noisy=data/noisy/combined/medium/cordis_h2020/en/test.jsonl tr_test_noisy=data/noisy/combined/medium/cordis_h2020/tr/test.jsonl --device cuda --output-dir results/predictions/cordis_v1.1.0
python scripts/predict_all.py --artifacts models/cordis_v1.1.0_knn/cordis_h2020/en_tr_direct --models hybrid_knn --sets en_validation tr_validation en_test tr_test cs_chunk_test cs_sentence_test en_test_noisy=data/noisy/combined/medium/cordis_h2020/en/test.jsonl tr_test_noisy=data/noisy/combined/medium/cordis_h2020/tr/test.jsonl --device cuda --output-dir results/predictions/cordis_v1.1.0
python scripts/predict_all.py --artifacts models/cordis_v0.5.0/cordis_h2020/en_tr_direct --models bilstm transformer --sets en_validation tr_validation en_test tr_test cs_chunk_test cs_sentence_test en_test_noisy=data/noisy/combined/medium/cordis_h2020/en/test.jsonl tr_test_noisy=data/noisy/combined/medium/cordis_h2020/tr/test.jsonl --device cuda --output-dir results/predictions/cordis_v1.1.0
```

Each `.npz` has a sidecar `.json` with the input SHA-256, record count and a digest of
every artifact file; the benchmark refuses dumps whose sidecar does not match the gold set.

Use `--limit 50` for a pilot. Kev, Jev and Laya are not in these full-set dumps; they are
scored on fixed samples (end of section 4).

## 4. Benchmark

```bash
python scripts/evaluate_ensemble.py --predictions results/predictions/cordis_v1.1.0 --members svm hybrid_knn bilstm transformer --baselines naive_bayes --tune-sets en_validation tr_validation --test-sets en_test tr_test cs_chunk_test cs_sentence_test en_test_noisy=data/noisy/combined/medium/cordis_h2020/en/test.jsonl tr_test_noisy=data/noisy/combined/medium/cordis_h2020/tr/test.jsonl --output-dir results/metrics/cordis_v1.1.0_ensemble
python scripts/make_figures.py --summary results/metrics/cordis_v1.1.0_ensemble/summary.json --predictions results/predictions/cordis_v1.1.0 --test-sets en_test tr_test cs_chunk_test --output-dir results/plots/cordis_v1.1.0
python scripts/translation_exceptions.py --summary results/metrics/cordis_v1.1.0_ensemble/summary.json --predictions results/predictions/cordis_v1.1.0 --sets en_test tr_test cs_chunk_test cs_sentence_test en_test_noisy=data/noisy/combined/medium/cordis_h2020/en/test.jsonl tr_test_noisy=data/noisy/combined/medium/cordis_h2020/tr/test.jsonl --artifact models/cordis_v1.1.0/cordis_h2020/en_tr_direct/svm/naltra.json
python scripts/run_benchmark.py --artifacts models/cordis_v1.1.0/cordis_h2020/en_tr_direct models/cordis_v1.1.0_knn/cordis_h2020/en_tr_direct models/cordis_v0.5.0/cordis_h2020/en_tr_direct --models naive_bayes svm hybrid_knn bilstm transformer --samples 200 --batch-size 64 --device cpu --output-dir results/benchmarks/cordis_v1.1.0
```

`translation_exceptions.py` reports the projects the prebuilt audit flagged and scores every
member with and without them; on 1.1.0 it finds none.

Validation is split by `pair_id` (EN and TR of a project stay together): val-A tunes each
member's threshold, val-B tunes the ensemble's vote count k and soft thresholds. Test sets
are only scored. Reported systems: each member, `hard_majority` (strict majority vote),
`hard_k` (tuned k-of-M), `hard_closed` (vote in the 586-node hierarchy, ancestor-consistent),
`soft`, `weighted_soft`, plus leave-one-out majority F1 per member. The benchmark refuses
tuning sets that share records, projects, pairs or text with a test set.

`summary.json["significance"]` holds paired project bootstrap results (10,000 resamples by default, grouped by `pair_id`,
95% percentile intervals, Holm-corrected within each family): each ensemble against the
best member chosen on val-A, and every system across conditions (the first test set vs
TR, code-switch and noisy sets, paired by `pair_id`). Run latency only on an idle machine.

The `run_benchmark.py` command above takes all three run directories and writes one
summary for the five trained families. The summary's `runtime` block records the package
versions and the `CUDA_VISIBLE_DEVICES`, `PYTORCH_NVML_BASED_CUDA_CHECK` and
`OMP_NUM_THREADS` settings of each run.

### Kev, Jev and Laya on fixed samples

Kev asks 473 yes/no questions per project, about 75 s per project on a GTX 1650, so it is
scored on 80 validation and 120 test project pairs selected by pair-id hash
(`scripts/sample_sets.py` keeps the pairs with the smallest SHA-256 of `pair_id`, so EN and
TR hold the same projects). Every member is rescored on the same files, so the comparison
is like for like. The 80 validation pairs are split by pair hash into 32 val-A pairs (64
bilingual rows: member thresholds and weights) and 48 val-B pairs (96 rows: ensemble
settings). Of the 473 direct labels, 249 have no positive project in the test sample and
304 none in the validation sample, so sample macro-F1 (at most 224/473 = 0.474) is not
comparable to the full sets. `data/samples/` is ignored by Git. Jev (TypeSafe's hosted
System One model) is scored on the same files; it needs `JEV_API_KEY` and
`JEV_API_BASE_URL=https://api.typesafe.ai` in `.env`. Laya (`pip install -r
requirements-laya.txt`) scores the same files from an artifact that `prepare_laya.py` builds
from the pinned checkpoint (a 678 MB download); at about 145 s per document on the GTX 1650
its 400 sample documents take about 16 h. Once its dumps exist, add `laya` to `--baselines` of
the sample `evaluate_ensemble.py` command, so it is scored without changing the votes. With the server of section 5 running:

```bash
python scripts/sample_sets.py --sets en_validation tr_validation --pairs 80 --output-dir data/samples/val80
python scripts/sample_sets.py --sets en_test tr_test --pairs 120 --output-dir data/samples/test120
python scripts/predict_all.py --artifacts models/cordis_v1.1.0/cordis_h2020/en_tr_direct --models naive_bayes svm kev jev --sets en_val_kev=data/samples/val80/en_validation.jsonl tr_val_kev=data/samples/val80/tr_validation.jsonl en_test_kev=data/samples/test120/en_test.jsonl tr_test_kev=data/samples/test120/tr_test.jsonl --device cuda --output-dir results/predictions/cordis_v1.1.0
python scripts/predict_all.py --artifacts models/cordis_v1.1.0_knn/cordis_h2020/en_tr_direct --models hybrid_knn --sets en_val_kev=data/samples/val80/en_validation.jsonl tr_val_kev=data/samples/val80/tr_validation.jsonl en_test_kev=data/samples/test120/en_test.jsonl tr_test_kev=data/samples/test120/tr_test.jsonl --device cuda --output-dir results/predictions/cordis_v1.1.0
python scripts/predict_all.py --artifacts models/cordis_v0.5.0/cordis_h2020/en_tr_direct --models bilstm transformer --sets en_val_kev=data/samples/val80/en_validation.jsonl tr_val_kev=data/samples/val80/tr_validation.jsonl en_test_kev=data/samples/test120/en_test.jsonl tr_test_kev=data/samples/test120/tr_test.jsonl --device cuda --output-dir results/predictions/cordis_v1.1.0
python scripts/prepare_laya.py --output-dir models/cordis_laya_v1.1.0/cordis_h2020/en_tr_direct
python scripts/predict_all.py --artifacts models/cordis_laya_v1.1.0/cordis_h2020/en_tr_direct --models laya --sets en_val_kev=data/samples/val80/en_validation.jsonl tr_val_kev=data/samples/val80/tr_validation.jsonl en_test_kev=data/samples/test120/en_test.jsonl tr_test_kev=data/samples/test120/tr_test.jsonl --device cuda --output-dir results/predictions/cordis_v1.1.0
python scripts/evaluate_ensemble.py --predictions results/predictions/cordis_v1.1.0 --members svm hybrid_knn bilstm transformer kev jev --baselines naive_bayes --tune-sets en_val_kev=data/samples/val80/en_validation.jsonl tr_val_kev=data/samples/val80/tr_validation.jsonl --test-sets en_test_kev=data/samples/test120/en_test.jsonl tr_test_kev=data/samples/test120/tr_test.jsonl --output-dir results/metrics/cordis_v1.1.0_sample
```

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

## Known limitations to state in the paper

- Kev and Jev dumps bind the service configuration, not the served weights: Kev ran
  `jaredpalmer/kev-0.8b` at `bf75a6a8848ea6960ff2ed108d9ed44c2941174f`, while Jev's
  `jev-latest` is a hosted alias whose weights can change between runs.

- Turkish is machine translation by `Helsinki-NLP/opus-mt-tc-big-en-tr` @
  `e539fc16a8a1a0ea5950eb339b595bfcce990e90`; 30,809 imported records keep the legacy alias
  `Helsinki-NLP/opus-mt-en-tr` in `translation_requested_model`, which
  `src/naltra/data/translation.py` normalises to that checkpoint.
- TR is release 1.1.0, on which every locally trained member is trained (for `cordis_v0.5.0`
  the head refit; XLM-R's encoder was fine-tuned in run v0.4.0, whose data is not recorded)
  and every member is evaluated: the 598 records the
  1.0.0 audit flagged were re-translated, and the prebuilt audit now finds no alignment
  mismatch or QA failure. On 1.0.0 those projects moved any member's micro-F1 by at most
  0.0011 (a historical figure; that run's output is not tracked).
- CORDIS labels are semi-automatic (silver), not human gold.
- `transformer` truncates at 512 subword tokens in `cordis_v0.5.0` (256 in the base run);
  Turkish documents are longer.
- The `xlm-roberta-base` revision was not pinned at training, and the `cordis_v0.5.0`
  artifact records `pretrained_revision` as null. Inference loads only the artifact's own
  files (the fine-tuned weights are its `model.safetensors`); a retrain may pull a different
  base snapshot.

## 6. Interactive results page

OOD (held-out humanities, NB and SVM retrained without it) runs first, so the page embeds
an OOD summary of the same release:

```bash
python scripts/run_ood.py --models naive_bayes svm --languages en tr --output-dir results/ood/cordis_v1.1.0
python scripts/build_dashboard.py --summary results/metrics/cordis_v1.1.0_ensemble/summary.json --predictions results/predictions/cordis_v1.1.0 --test-sets en_test tr_test cs_chunk_test cs_sentence_test en_test_noisy=data/noisy/combined/medium/cordis_h2020/en/test.jsonl tr_test_noisy=data/noisy/combined/medium/cordis_h2020/tr/test.jsonl --latency results/benchmarks/cordis_v1.1.0/benchmark_summary.json --ood results/ood/cordis_v1.1.0/summary.json --sample-summary results/metrics/cordis_v1.1.0_sample/summary.json --output docs/results/index.html
```

`run_ood.py` replaces a previous `run_ood.py` output in `--output-dir`. It refuses any other
existing directory unless `--overwrite` is given, and always refuses an output directory that
overlaps `--data-dir` or contains the repository. OOD sets aside the 266 EN (and 266 TR) test
projects that mix humanities with other roots: 4,351 in-distribution and 94 pure humanities
test projects are scored (`counts` in its `summary.json`).

`--sample-summary` adds the 120-pair sample table of section 4, where Kev and Jev are scored
with every other member. Its `--latency` input is the one combined summary that the section 4 `run_benchmark.py`
command writes for all five trained families, `results/benchmarks/cordis_v1.1.0/`.
The page is one self-contained HTML file (no build step, no server). Open it locally, or
enable GitHub Pages for the `docs/` folder (Settings → Pages → Deploy from branch, `/docs`)
and it is served at `https://<owner>.github.io/NALTRA/results/`. It refuses to build from
dumps the summary did not score or from a different taxonomy.
