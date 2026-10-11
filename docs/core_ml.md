# Model usage

Use the shared prediction pipeline to load one model or combine multiple models.
Install dependencies using the [quick start](../README.md#quick-start), or use the
[pinned environment](reproducibility.md#environment) for benchmark artifacts.

## Setup and training

Install original artifacts according to the [model file guide](../models/README.md).
The [runbook](runbook.md#2-train) contains full training/refit commands. Kev, Jev,
and Laya are zero-shot members and are not trained on CORDIS.

For a bounded CPU integration check using the restored dataset:

```bash
python scripts/train_all.py --prebuilt-release --smoke --models bilstm transformer --device cpu --output-dir models/smoke
```

Smoke mode uses small architectures and a training-only tokenizer; its artifacts
are not benchmark models. Hybrid kNN smoke mode still needs the e5 encoder.

Training never prepares data. It audits inputs, trains on direct targets, builds
vocabularies from training only, and selects checkpoints using validation loss.
Test inputs are audited but not used for fitting. Neural training uses AdamW,
gradient clipping, early stopping, and the lowest-validation-loss checkpoint.
CUDA uses FP16 autocast/scaling; CPU uses ordinary precision.

A head refit freezes the existing encoder, vocabulary, and label order and trains
only the sigmoid classifier. `cordis_v0.5.0` uses 512-token inputs and unweighted
BCE after this refit. [Evaluation](evaluation.md#how-the-cordis_v050-neural-members-were-built)
documents the candidate selection and original encoder provenance.

Configurations live in `configs/models/` and `configs/training.yaml`.
The default training families are BiLSTM and Transformer; name `--models`
explicitly for another selection. Saved configuration, history, dataset identity,
seeds, and framework/device details describe each run. Existing artifacts require
a fresh output directory or explicit `--overwrite`. Each run rewrites
`training_summary.json` in its output directory.

## Prediction and artifacts

Models implement `BaseNALTRAModel` for training, prediction, batching, saving, and
loading. Neural constructors accept nested configuration overrides and an optional
`taxonomy_path`.

`PredictionResult.label_scores` contains probabilities for every supported direct
label. `labels` in `PredictionResult` contains threshold-selected direct labels and may be empty. `metadata`
records label space, taxonomy, calibration status and OOD configuration. Probabilities/latencies must be finite;
duplicate selected labels are rejected. Serialization preserves these fields.
Hard ensemble scores are explicitly marked as vote fractions rather than class
probabilities in metadata.

Artifacts live under `<output>/<dataset>/<track>/<model>/`. CORDIS tracks are `en_direct`, `tr_direct`, or `en_tr_direct`; the artifact records `target_field`, language selection, and dataset manifests. `naltra.json` records
configuration, label order, taxonomy/label-map hashes, history, OOD threshold and
provenance. BiLSTM adds weights/vocabulary; Transformer saves Hugging Face model
and tokenizer files. Loading checks family and taxonomy compatibility and uses
local files only. The caller's device overrides the producing machine.

Built wheels include the default configs, taxonomy and label map, so model construction,
artifact loading and the prediction pipeline also work outside the source checkout.
Dataset preparation and experiment scripts use a source checkout and the
[runbook](runbook.md).

```python
from naltra.models.bilstm import BiLSTMModel
from naltra.pipeline.prediction import PredictionPipeline
from naltra.schemas.prediction import LanguageInfo

model = BiLSTMModel({"device": "cpu"})
model.load("models/cordis_v0.5.0/cordis_h2020/en_tr_direct/bilstm")
pipeline = PredictionPipeline(model=model)
result = pipeline.predict("Türkiye'de yeni bilimsel araştırmalar yapılıyor.")
batch = pipeline.predict_batch(
    ["Science research", "Bilimsel araştırma"],
    languages=[LanguageInfo("en"), LanguageInfo("tr")],
)
print(result.to_dict())
```

The pipeline normalizes whitespace, detects language, predicts, applies optional
threshold overrides, resolves hierarchy paths and computes OOD. Returned text is
normalized and batch order is preserved. Batch `latency_ms` is total pipeline
time divided by batch size, not independently measured record latency. Direct
model predictions return `language: und`; use the pipeline or known benchmark
metadata for language information.

Lingua works offline after installation and is restricted to EN/TR. Very short,
numeric or low-confidence text returns `und`. Mixed detection requires confident
spans containing at least two words per language and at least 20% of detected
words for each language. Mixed detection is experimental; benchmark slicing
should use known record metadata.

Required ancestors appear in root-to-label `hierarchy_paths` without fabricated
probabilities. Direct `labels` remain unchanged for classification metrics;
hierarchical metrics expand ancestors. The taxonomy's `require_parent_predictions`
policy is fulfilled by ancestor paths.

OOD score is `1 - max(label_scores)`, with higher scores meaning lower confidence.
After checkpoint selection, training sets the threshold to the 95th percentile
of ID validation scores. A head refit (`--refit-head-from`) clears it, because the
threshold no longer fits the new head, so the reported `cordis_v0.5.0` BiLSTM and
Transformer artifacts store `ood_threshold: null` and report OOD as `disabled`; the
NB, SVM and hybrid kNN artifacts keep theirs. The pipeline flags scores strictly above
the threshold. This is a confidence baseline, not a calibrated OOD performance claim. OOD predictions
retain their classification labels; no fake OOD taxonomy label is created.
Missing thresholds are explicitly marked disabled. Callers may supply a custom
text OOD detector or explicit score threshold. Calibration and explanation
callables are hooks; absent explainers produce no fabricated attribution.
A custom calibrator invalidates the original OOD threshold; supply a threshold
fitted on calibrated validation scores or detection remains disabled.

## Ensembles

To use the published decisions with the exact artifacts that were evaluated, pass the
offline summary to the classification CLI:

```bash
python scripts/classify.py --artifacts models/cordis_v1.1.0/cordis_h2020/en_tr_direct --models svm --benchmark-settings results/metrics/cordis_v1.1.0_ensemble/summary.json --text "Laser sensors measure light."
```

For an ensemble, `--models` must match the summary's voting members exactly; they must
all be available under `--artifacts`. Choose `--method hard`, `hard_k`, `soft`, or
`weighted_soft`. The methods `hard_k` and `weighted_soft` require
`--benchmark-settings`. A mismatched artifact or taxonomy fails before inference.
Outputs record the settings-file SHA-256 in `threshold_source`. Without the flag,
the CLI retains the artifact thresholds and the default hard/soft voting behavior.

Soft/weighted-soft voting uses complete per-label scores, including values below
selection thresholds; a result without `label_scores` falls back to the scores of its
selected labels and treats omissions as zero. Hard voting
uses selected-label membership and defaults to strict-majority semantics; an explicit
`vote_count` enables k-of-M voting. Declared
dataset, track and supported-label spaces must match. Missing required models
fail explicitly; they are never silently removed from the vote.

```python
from naltra.models.bilstm import BiLSTMModel
from naltra.models.transformer import TransformerModel
from naltra.pipeline.prediction import PredictionPipeline
from naltra.pipeline.ensemble import EnsembleConfig

model = BiLSTMModel({"device": "cpu"})
model.load("models/cordis_v0.5.0/cordis_h2020/en_tr_direct/bilstm")
transformer = TransformerModel({"device": "cpu"})
transformer.load("models/cordis_v0.5.0/cordis_h2020/en_tr_direct/transformer")
config = EnsembleConfig(
    method="weighted_soft",
    required_models={"bilstm", "transformer"},
    weights={"bilstm": 1.0, "transformer": 1.0},
)
ensemble = PredictionPipeline(
    models={"bilstm": model, "transformer": transformer}, ensemble_config=config,
)
```

Ensembles have no automatically fitted OOD threshold. Fit a threshold on that
ensemble's ID validation scores and pass it explicitly; do not reuse a component
threshold. Hard-vote fractions are not class probabilities, so confidence OOD
is disabled for hard voting; use a custom detector.

For a JSONL batch file containing a non-empty `text` field per line, replace
`--text` with `--input path/to/input.jsonl`. The CLI prints one JSON result per input.

The CLI accepts one common parent artifact directory for an ensemble. The published
members originally live under three run directories. To use them together, place
copies of their complete family folders under a new common parent while preserving
all file contents; the [model guide](../models/README.md) lists the originals.
The Python pipeline can load members directly from their individual paths.

## Kev and Jev service adapters

Kev and Jev read `KEV_*` / `JEV_*` keys and URLs from ignored `.env` at inference
time; they are never persisted in artifacts. Each model YAML's `contract` sets
`configured`, the relative `request_path`, `text_field`, dotted `scores_path`,
`auth_header` and `auth_prefix`. `supported_labels` (empty means every direct label)
and `label_map` fix the canonical label space; every response must answer every
supported label with a finite probability.

Adapters have configurable timeouts and fail clearly on transport errors, invalid
JSON, unsupported labels or missing answers. They do not automatically retry POST
requests or follow redirects. Service `train()` reports unsupported local training.
Mock transports cover both members.

## Local multilingual Laya

Install the optional pinned SDK from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-laya.txt
```

`LayaModel()` loads the multilingual checkpoint lazily on first inference. The
checkpoint commit and SDK version are pinned in `configs/models/laya.yaml`; the
default covers all 473 supported direct taxonomy labels. English, Turkish and
mixed-language inputs use the same checkpoint. No API key or payment is required.

```python
from naltra.models.laya import LayaModel
from naltra.pipeline.prediction import PredictionPipeline

model = LayaModel({"device": "cuda"})
pipeline = PredictionPipeline(model=model)
result = pipeline.predict("This project develops solar panels and battery storage.")
print([(label.label, label.score) for label in result.labels])
```

Each label is a separate two-option `choice` question, asked in English for every
input language. Option `A` names the canonical scientific topic; option `B` is "other
unrelated research topics".
`P(A)` becomes the direct-label score. Labels are independent and their probabilities
do not sum to one. Two-option choice follows the upstream workaround for the
documented `noul` label bias.
Eight questions and one document per inference batch bound GPU memory. The
1,024-token total budget includes a 256-token question budget; longer input is
truncated. These are pretrained, uncalibrated scores, not a model fine-tuned on
CORDIS. The 0.5 cutoff in `configs/models/laya.yaml` is only the default for direct
use; the benchmark tunes Laya's threshold on val-A like every other member's. A
teammate's eight-record integration check at that default selected too many labels; it
verified execution, not accuracy. `train()` deliberately reports unsupported: Laya is
scored zero-shot.

### Benchmark route

Prepare Laya using `scripts/prepare_laya.py`, then score the fixed EN/TR samples
using the [sample commands](runbook.md#kev-jev-and-laya-on-fixed-samples).
The prepared folder includes pinned checkpoint files, SHA-256 values, and exact
label questions. Reload verifies hashes and uses local files only.

Once all sample dumps exist, add `laya` to the sample evaluator's non-voting
members (`--baselines naive_bayes laya`). Its threshold is tuned on val-A, but
it does not change the voting members, and the flag does not make it a course
baseline: its role stays `new`. The recorded 120-pair test result is
0.068 English / 0.046 Turkish micro-F1, using a threshold of 0.98 tuned on val-A,
the top of the threshold grid ([sample protocol](evaluation.md#sample-protocol-kev-jev-laya)).

### Direct validation route: `evaluate_all.py`

The teammates' `scripts/evaluate_all.py` evaluates BiLSTM, XLM-R and Laya artifacts
directly, with a resumable prediction cache. It audits the data with the full validator,
so it needs a release rebuilt from the raw archive and refuses the shipped one
([release validation](dataset.md#release-validation)); no reported number comes from it.
Create a portable Laya artifact and check two records from each validation track:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_all.py --models laya --prepare-laya --split validation --code-switch sentence_mix chunk_mix --max-records 2 --device cuda --batch-size 1 --model-dir models/cordis_laya_v0.1.1/cordis_h2020/en_tr_direct --output-dir results/metrics/cordis_laya_v0.1.1_integration_validation
```

Use a new output directory if that check already exists. `--prepare-laya` refuses
an existing artifact and is limited to bilingual validation. It binds the audited
release without training and copies weights, encoder/tokenizer configs, exact
questions and checksums into the ignored model folder.

Once the subset check is satisfactory, tune on full clean validation:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_all.py --models laya --split validation --tune-threshold --resume --device cuda --batch-size 1 --model-dir models/cordis_laya_v0.1.1/cordis_h2020/en_tr_direct --output-dir results/metrics/cordis_laya_v0.1.1_validation_tuned
```

Then reuse that report with `--thresholds-file` and `--code-switch sentence_mix
chunk_mix --code-switch-only --split validation` for the mixed-language check.
In this route the cutoff is selected only after both validation languages finish;
subset reports cannot supply it. Classify all labels even when only a few are
selected; partial top-k results cannot be compared fairly with the neural models.

This run scores 9,422 documents against 473 independent label questions. On the
RTX 4050 laptop, the measured two-document speed check took 14.5 seconds;
budget roughly 18–24 hours for full validation, depending on document lengths.
Larger question batches did not improve measured throughput. `--resume` commits each
completed batch to the ignored `prediction_cache.sqlite3` beside the metrics report.
Repeat the same command after an interruption; add `--overwrite` if a failed summary
already exists. Changed artifacts, inputs, settings, dependencies or evaluation code are
rejected instead of silently reusing incompatible predictions.

Upstream: [model card](https://huggingface.co/convaiinnovations/laya) and
[SDK](https://github.com/NandhaKishorM/laya).
