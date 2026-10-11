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
Laya has a recorded 120-pair sample result as a non-voting baseline; its original
artifact and prediction dumps are also needed for independent verification.

## Environment

Run from a source checkout. The package supports Python 3.11+. The recorded
experiment environment used Python 3.13.1 on Windows; `requirements.txt` needs
Python 3.12+ and pins direct, selected runtime, and development dependencies.

Create and activate an environment as shown in the [quick start](../README.md#quick-start),
then install the recorded versions:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For GPU inference, match the CUDA-capable PyTorch build recorded by the experiment
(`2.14.1+cu126` for the reported GPU runs) and a compatible driver. CPU replay of
saved prediction matrices needs no CUDA. Dependency pins do not reproduce
hardware, drivers, or every transitive package automatically.

Keep `.env` credentials local. Recalculating metrics from archived Kev/Jev
predictions needs no live API credentials.

## Install the handoff

Request the **original artifacts used for the reported experiments**, preserving
paths and file contents. Newly retrained artifacts form a new release.

| Bundle | Installation location | Inventory |
| --- | --- | --- |
| Trained models | `models/` | [Five model folders and dependencies](../models/README.md) |
| Prediction dumps | `results/predictions/cordis_v1.1.0/` | [68 NPZ/JSON pairs](../results/README.md#prediction-dumps) |
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

The dashboard also needs the recorded latency and OOD summaries. See
[runbook section 6](runbook.md#6-interactive-results-page).
Inference latency must be remeasured on an idle machine and reported with its
hardware and thread settings; it cannot be expected to match across machines.

## Rerun inference

Follow [prediction generation](runbook.md#3-dump-predictions), directing new dumps
to a fresh run folder. Compare label columns, input/artifact fingerprints, and
score matrices against the archived dumps before comparing thresholded metrics.

BiLSTM and Transformer load their saved files locally. Hybrid kNN needs its pinned
multilingual-e5 encoder. Kev needs its recorded weights and server implementation.
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
