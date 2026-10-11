# Reproducibility

Reproduction has three distinct levels. Choose the level needed before downloading
weights or starting a training run.

| Level | Needed | What it verifies |
| --- | --- | --- |
| Recalculate results | Dataset release, original NPZ/JSON predictions, evaluation code, frozen settings | Metrics, threshold tuning, ensemble comparisons, figures, and dashboard |
| Rerun inference | Exact trained artifacts, inputs, matching dependencies, and required encoders/services | Model outputs on the same inputs |
| Retrain | Original training data, initial checkpoints, commands, configurations, and seeds | The training procedure; floating-point outputs can differ across environments |

## Release status

The source archive, compressed datasets, manifests, split map, taxonomy, review
files, and result summaries are committed. The documentation assumes shared
artifacts are installed at the paths below.

The model/prediction handoff, download links, and independent verification are
pending. No uploaded bundle or successful artifact replay is claimed here.
Laya has a recorded 120-pair sample result (scored, never votes); its original
artifact and prediction dumps are also needed for independent verification.

## Environment

Run from a source checkout. The package supports Python 3.11+. The recorded
experiment environment used Python 3.13.1 on Windows; `requirements.txt` needs
Python 3.12+ and pins direct, selected runtime, and development dependencies.

Use a git clone (`git clone https://github.com/Rxd3/NALTRA`), not a ZIP download:
summaries record the commit with `git describe --always --dirty`, manifests with
`git rev-parse`. Create and activate an environment as shown in the
[quick start](../README.md#quick-start), then install the recorded versions:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu   # CPU-only machines: skips the CUDA build
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The CPU build satisfies the `torch==2.14.1` pin in `requirements.txt`. For GPU steps,
match the CUDA-capable PyTorch build recorded by the experiment (`2.14.1+cu126` for the
reported GPU runs) and a compatible driver:

```bash
python -m pip install --no-deps --force-reinstall torch==2.14.1+cu126 --index-url https://download.pytorch.org/whl/cu126
```

CPU replay of saved prediction matrices needs no CUDA. Dependency pins do not reproduce
hardware, drivers, or every transitive package automatically. The commands here run in
PowerShell and bash alike; in Windows PowerShell 5.1 run them one per line (no `&&`), and
`make` targets need Linux/macOS. On Windows, run the test suite one file per process
([why](../CONTRIBUTING.md#development-setup)).

Keep `.env` credentials local. Recalculating metrics from archived Kev/Jev
predictions needs no live API credentials.

## Install the handoff

Request the **original artifacts used for the reported experiments**, preserving
paths and file contents. Newly retrained artifacts form a new release.

| Bundle | Installation location | Inventory |
| --- | --- | --- |
| Trained models | `models/` | [Five model folders and dependencies](../models/README.md) |
| Prediction dumps | `results/predictions/cordis_v1.1.0/` | [72 NPZ/JSON pairs](../results/README.md#prediction-dumps) |
| OOD models | `results/ood/cordis_v1.1.0/models/` | Separate Naive Bayes and SVM artifacts |
| Dataset snapshot, if supplied | `data/` | Existing release, manifests, split map, and optional generated samples/noisy copies |

Keep artifact archives separate from credentials and environments. Archives should
include complete model folders, rather than weights alone.

Restore the committed inputs:

```bash
python scripts/unpack_data.py
python scripts/validate_datasets.py --prebuilt-release
python scripts/prepare_data.py --dataset noisy --prebuilt-release
python scripts/sample_sets.py --sets en_validation tr_validation --pairs 80 --output-dir data/samples/val80
python scripts/sample_sets.py --sets en_test tr_test --pairs 120 --output-dir data/samples/test120
```

## Verify file identity

The artifact publisher should provide `SHA256SUMS` and a release manifest containing
archive filenames, byte sizes, checksums, download locations, release versions,
the original experiment commit, and the code commit used for replay. These are
release deliverables to be supplied; no generated manifest is included yet.

Check an archive against the publisher's SHA-256 before extracting it:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath .\naltra-models-v1.zip
```

On Linux/macOS, use `sha256sum` or `shasum -a 256`.
Dataset unpacking validates split checksums and sizes. Original model file
fingerprints are recorded in the benchmark summaries under
`dumps[*].artifact_files`; `classify.py --benchmark-settings` checks loaded
artifact identities. Sidecars bind the prediction dumps to inputs and artifacts.

Do not rewrite `naltra.json`, tokenizers, or checksums to resolve a mismatch.
Identify the correct release instead.

## Recalculate the reported results

Follow [runbook section 4](runbook.md#4-benchmark) using the installed prediction
dumps. It includes separate commands for the full benchmark and the fixed samples.
Write new summaries into `results/metrics/replay_full/` and
`results/metrics/replay_sample/`, then compare them to the committed summaries.

Check member thresholds, vote settings, per-system metrics, and bootstrap
differences. Run metadata and paths may differ; compare the experimental values.
For archived score matrices under matching dependencies, expect matching decisions
and metrics, allowing small numerical differences in floating-point summaries.

With the dumps installed at `results/predictions/cordis_v1.1.0/`,
`python scripts/extra_analyses.py` rebuilds the coverage, pairwise, error, robustness and
screen files listed in [results sources](results.md#sources) (`mtquality` runs only when
named and needs the pinned TR -> EN model). It checks the dumps against the hashes the
committed summaries recorded and rewrites those files in place, so compare them with
`git diff`.

The dashboard also needs the recorded latency and OOD summaries. See
[runbook section 6](runbook.md#6-interactive-results-page).
Inference latency must be remeasured on an idle machine and reported with its
hardware and thread settings; it cannot be expected to match across machines.

## CPU check: Naive Bayes and SVM

Without the shared artifacts, a clone can retrain the two classical members on CPU and
regenerate their full-set and sample rows, intervals and near-OOD results (about 45 min on
a busy laptop; about 1.5 GB of data, models and results besides the environment). Restore
the committed inputs and samples first ([above](#install-the-handoff)), then run:

```powershell
$env:LOKY_MAX_CPU_COUNT = "4"                            # bash: export LOKY_MAX_CPU_COUNT=4
python scripts/train_all.py --datasets cordis_h2020 --prebuilt-release --models naive_bayes --device cpu --output-dir models/cpu_check   # about 4 min
python scripts/train_all.py --datasets cordis_h2020 --prebuilt-release --models svm --device cpu --output-dir models/cpu_check           # about 7-13 min
# Windows: if a run ends with exit 139 / -1073741819 / 0xC0000005 and models/cpu_check/cordis_h2020/en_tr_direct/<family>/naltra.json is missing, re-run that family's line
python scripts/predict_all.py --artifacts models/cpu_check/cordis_h2020/en_tr_direct --models naive_bayes svm --sets en_validation tr_validation en_test tr_test cs_chunk_test cs_sentence_test en_test_noisy=data/noisy/combined/medium/cordis_h2020/en/test.jsonl tr_test_noisy=data/noisy/combined/medium/cordis_h2020/tr/test.jsonl --device cpu --output-dir results/predictions/cpu_check
python scripts/evaluate_ensemble.py --predictions results/predictions/cpu_check --members naive_bayes svm --tune-sets en_validation tr_validation --test-sets en_test tr_test cs_chunk_test cs_sentence_test en_test_noisy=data/noisy/combined/medium/cordis_h2020/en/test.jsonl tr_test_noisy=data/noisy/combined/medium/cordis_h2020/tr/test.jsonl --output-dir results/metrics/cpu_check
python scripts/predict_all.py --artifacts models/cpu_check/cordis_h2020/en_tr_direct --models naive_bayes svm --sets en_val_kev=data/samples/val80/en_validation.jsonl tr_val_kev=data/samples/val80/tr_validation.jsonl en_test_kev=data/samples/test120/en_test.jsonl tr_test_kev=data/samples/test120/tr_test.jsonl --device cpu --output-dir results/predictions/cpu_check
python scripts/evaluate_ensemble.py --predictions results/predictions/cpu_check --members naive_bayes svm --tune-sets en_val_kev=data/samples/val80/en_validation.jsonl tr_val_kev=data/samples/val80/tr_validation.jsonl --test-sets en_test_kev=data/samples/test120/en_test.jsonl tr_test_kev=data/samples/test120/tr_test.jsonl --output-dir results/metrics/cpu_check_sample
python scripts/run_ood.py --models naive_bayes svm --languages en tr --output-dir results/ood/cpu_check
```

The two families train in separate runs so that a native crash or a joblib "A task has
failed to un-serialize" error costs only one of them ([runbook section 2](runbook.md#2-train));
each run rewrites `training_summary.json`, so check its `"success"` status before the next.
`evaluate_ensemble.py` needs two voting members, so this check lets Naive Bayes vote;
thresholds are tuned per member and every bootstrap uses the same seed, so the Naive Bayes
and SVM rows and intervals equal the committed full-set and sample ones (checked with the
`requirements.txt` pins and earlier output folder names; the folders were renamed to
`cpu_check` afterwards, and the other flags are unchanged). Which fields must match and
which differ by design are listed in
[results/README.md](../results/README.md#checking-the-cpu-re-run).

## Rerun inference

Follow [prediction generation](runbook.md#3-dump-predictions), directing new dumps
to a fresh run folder. Compare label columns, input/artifact fingerprints, and
score matrices against the archived dumps before comparing thresholded metrics.

| Member or step | Needs beyond a CPU clone |
| --- | --- |
| `bilstm`, `transformer` | The `cordis_v0.5.0` artifacts (loaded locally), or GPU retraining with an `xlm-roberta-base` download |
| `hybrid_knn` | Its pinned `intfloat/multilingual-e5-base` encoder (1.1 GB) and, in practice, a GPU |
| `kev` | A WSL2 Kev server with the recorded `jaredpalmer/kev-0.8b` weights ([runbook](runbook.md#5-kev-server-wsl2-ubuntu)) and `KEV_API_KEY`, `KEV_API_BASE_URL` in `.env`; about 75 s per project on a 4 GB GTX 1650, hence the 120-pair sample |
| `jev` | `JEV_API_KEY` (paid hosted API) and `JEV_API_BASE_URL` in `.env` (copy `.env.example`; never commit `.env`) |
| `laya` | `pip install -r requirements-laya.txt`, the 678 MB pinned checkpoint (`prepare_laya.py`) and, in practice, a GPU: about 145 s (EN) to 200 s (TR) per document on the GTX 1650, about 19 h for the 400 sample documents |
| `run_benchmark.py` (latency) | The artifacts of all five trained families and an idle machine |

Jev's hosted `jev-latest` alias can change; archived predictions preserve the
reported experiment even when a live rerun differs.

## Training and data provenance

Full training replay still requires the owner to provide:

- Original `cordis_v0.4.0` BiLSTM/Transformer artifacts used before the head refit.
- Exact original training dataset versions and manifests; XLM-R encoder fine-tuning
  data is not recorded in the current release documentation.
- The Turkish 1.1.0 repair script/notebook, settings, affected record IDs, and logs
  for the 598-record repair.
- Original commands, resolved configurations, training summaries/history, seeds,
  dependency details, Git commits, and hardware.
- Kev server commit/launch settings and any available Jev model-version/run-date details.

Result-level gaps (a summary without a commit field, counts computed from committed data
without a committed script) are listed under [recorded outputs](evaluation.md#recorded-outputs).
Sharing final weights enables inference; it does not recover missing training
lineage. Standard translation produces a different Turkish release. Use a separate
checkout and a new release version for fresh dataset preparation or retraining.

## Publisher checklist

1. Freeze the reviewed code and record both producing and replay commits.
2. Package original model folders and prediction pairs without modifying files.
3. Publish versioned archives, a checksum list, and a release manifest; add links here.
4. Test download, verification, installation, and metric replay from a fresh checkout.
5. Record which reproduction levels passed and any remaining provenance gaps.

Use a fixed archive version rather than replacing files behind an existing release
link. A public archive such as Zenodo can be the primary release location; Google
Drive can serve as a download mirror.
