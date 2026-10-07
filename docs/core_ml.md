# Core ML operation and integration

BiLSTM and Transformer train independently on prepared records and save locally
reloadable artifacts. Jev/Laya are configurable inference adapters; their live
services require actual provider contracts. Classical models, dashboard wiring,
expanded evaluation, calibration, and attribution remain separate work.

## Setup and training

Activate the project environment and install the updated dependencies:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

New dependencies include `httpx` and `lingua-language-detector`. CUDA training
requires a compatible PyTorch build; CPU is supported. Dataset manifests include
dependency-file hashes, so changes require genuine regeneration, not hash edits:

```powershell
python scripts/prepare_data.py --dataset all --download
python scripts/validate_datasets.py
```

Preparation verifies the locked CORDIS raw CSVs and publishes manifest schema 1.1.0. Translation pins the OPUS-MT checkpoint and versions its cache with the generation code.
Preserve the complete data release and model artifacts with experiments; final
release data should be generated from reviewed code as described in [dataset
provenance](dataset.md).

Run bounded offline smoke training or full experiments:

```powershell
python scripts/train_all.py --smoke --device cpu --output-dir models/smoke
python scripts/train_all.py --device auto
python scripts/train_all.py --models bilstm --datasets cordis_h2020 --languages en --device cpu
python scripts/train_all.py --models transformer --datasets cordis_h2020 --languages en tr --device cuda --batch-size 2 --accumulation 8
python scripts/train_all.py --datasets cordis_h2020 --languages en tr --device auto --output-dir models/cordis_v0.4.0
```

`configs/training.yaml` selects CORDIS English and Turkish jointly by default. Use `--languages en` for the original-language baseline. The loader audits each manifest and checks project/content partition isolation, complete direct-label training coverage, translation provenance, and aligned source fidelity before initializing any model. Missing Turkish records fail a bilingual request explicitly.

Model YAML files control architecture and training. Overrides include `--epochs`, `--batch-size`, `--accumulation`, `--device`, `--output-dir`, `--config`, and `--smoke-records`. Existing artifacts require a new output directory or explicit `--overwrite`. Failures produce a nonzero exit status and are recorded in `training_summary.json`. Classical placeholders and inference services are not locally trainable.

Training never prepares data. Test files are audited for integrity but never used for fitting, vocabulary construction, or checkpoint selection. Noise and code-switch records remain evaluation-only. Historical SIB-200 OOD folds are not CORDIS training inputs; a CORDIS held-out-domain protocol must be defined separately.

Smoke mode uses at most 128 training and compatible validation records, one epoch
unless overridden, smaller BiLSTM dimensions, and a tiny randomly initialized
Transformer with a training-only tokenizer. Smoke artifacts have a `_smoke`
track suffix and `smoke: true`; they are not benchmark models. Validation records
with labels absent from the bounded training subset are excluded. Increase
`--smoke-records` if no compatible validation records remain.

Both CORDIS models explicitly use `target_field: labels_direct` to train multi-hot direct-label targets with binary cross-entropy with
logits. Labels are sorted canonical IDs from training. BiLSTM preserves Unicode
and casing, builds its vocabulary only from training, uses learned random
embeddings, and packs sequences so padding does not affect recurrent state. This
baseline does not use externally aligned bilingual embeddings. Transformer uses
XLM-RoBERTa with a multi-label classification head.

Training uses AdamW, gradient clipping, and early stopping after two epochs
without validation-loss improvement. The lowest-validation-loss checkpoint is
restored. Without validation, the final epoch is used and OOD remains disabled.
Transformer defaults use batch size 2, eight accumulation steps, and gradient
checkpointing. CUDA uses FP16 autocast/scaling; CPU uses ordinary precision.
Seeds, resolved configuration, framework versions and device details are saved;
bitwise reproducibility across hardware/versions is not guaranteed.

Full Transformer training needs an initial pretrained-weight download or local
checkpoint. For local weights, set `architecture.pretrained_name` to that directory
and `local_files_only: true`. An optional `revision` pins Hub weights; resolved
commit metadata is recorded when available. Saved artifacts reload offline.

## Prediction and artifacts

The five `BaseNALTRAModel` methods are unchanged. Neural constructors accept
nested config overrides and optional `taxonomy_path`; Transformer also accepts
an injected tokenizer/network pair for local fixtures. Existing wrappers work.

`PredictionResult.label_scores` contains probabilities for every supported direct
label. `labels` in `PredictionResult` contains threshold-selected direct labels and may be empty. `metadata`
records label space, taxonomy, calibration status and OOD configuration. New
fields have defaults for compatibility. Probabilities/latencies must be finite;
duplicate selected labels are rejected. Serialization includes the new fields.
Hard ensemble scores are explicitly marked as vote fractions rather than class
probabilities in metadata.

Artifacts live under `<output>/<dataset>/<track>/<model>/`. CORDIS tracks are `en_direct`, `tr_direct`, or `en_tr_direct`; the artifact records `target_field`, language selection, and dataset manifests. Older artifacts use a different taxonomy and must be retained as historical runs. `naltra.json` records
configuration, label order, taxonomy/label-map hashes, history, OOD threshold and
provenance. BiLSTM adds weights/vocabulary; Transformer saves Hugging Face model
and tokenizer files. Loading checks family and taxonomy compatibility and uses
local files only. The caller's device overrides the producing machine.

```python
from naltra.models.bilstm import BiLSTMModel
from naltra.pipeline.prediction import PredictionPipeline
from naltra.schemas.prediction import LanguageInfo

model = BiLSTMModel({"device": "cpu"})
model.load("models/cordis_v0.4.0/cordis_h2020/en_tr_direct/bilstm")
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
of ID validation scores. The pipeline flags scores strictly above it. This is a
confidence baseline, not a calibrated OOD performance claim. OOD predictions
retain their classification labels; no fake OOD taxonomy label is created.
Missing thresholds are explicitly marked disabled. Callers may supply a custom
text OOD detector or explicit score threshold. Calibration and explanation
callables are hooks; absent explainers produce no fabricated attribution.
A custom calibrator invalidates the original OOD threshold; supply a threshold
fitted on calibrated validation scores or detection remains disabled.

## Ensembles

Soft/weighted-soft voting uses complete probabilities, including values below
selection thresholds. Legacy results fall back to selected scores. Hard voting
uses selected-label membership and preserves strict-majority semantics. Declared
dataset, track and supported-label spaces must match. Missing required models
fail explicitly; they are never silently removed from the vote.

```python
from naltra.models.transformer import TransformerModel
from naltra.pipeline.ensemble import EnsembleConfig
from naltra.utils.config import load_yaml

transformer = TransformerModel({"device": "cpu"})
transformer.load("models/cordis_v0.4.0/cordis_h2020/en_tr_direct/transformer")
config = EnsembleConfig(**load_yaml("configs/ensemble_neural.yaml")["ensemble"])
ensemble = PredictionPipeline(
    models={"bilstm": model, "transformer": transformer}, ensemble_config=config,
)
```

Ensembles have no automatically fitted OOD threshold. Fit a threshold on that
ensemble's ID validation scores and pass it explicitly; do not reuse a component
threshold. Hard-vote fractions are not class probabilities, so confidence OOD
is disabled for hard voting; use a custom detector. The original six-family
configuration becomes usable when all components are available.

## Completing live Jev/Laya integration

Neither name identifies a documented provider in this repository. Obtain actual
service documentation and sample responses before enabling them:

1. Confirm endpoint, authentication, request and response formats, label vocabulary
   and score semantics. Current transport supports synchronous JSON POST with one
   text field and a label-to-probability response object. Other formats require a
   provider-specific injected client implementing `predict(text)` and returning
   `{"scores": {provider_label: probability}}`.
2. Set `JEV_API_KEY` / `JEV_API_BASE_URL` or equivalent `LAYA_*` values in ignored
   `.env`. Keys and URLs are read at inference time and never persisted in artifacts.
3. Fill the model YAML's `contract`: `configured: true`, relative `request_path`,
   `text_field`, dotted `scores_path`, `auth_header` and `auth_prefix`. Placeholder
   names are examples, not claims about a real API.
4. Set `supported_labels` to the experiment's complete canonical label space and
   `label_map` to provider-to-canonical IDs. Every response must include finite
   probabilities for all supported labels. Partial top-k lists and ranking scores
   require a documented provider-specific conversion rather than guessed values.
5. Set `enabled: true`, pass the YAML to `JevModel(config=...)` or
   `LayaModel(config=...)`, and verify a real sample through the pipeline.

Adapters have configurable timeouts and fail clearly on transport errors, invalid
JSON, unsupported labels or missing scores. They do not automatically retry POST
requests or follow redirects. Service `train()` reports unsupported local training.
Mock transports cover both families; live behavior and provider-owned training
cannot be verified without real provider details and credentials.

Naive Bayes/SVM, dashboard integration, expanded evaluation, calibration and
attribution remain with their existing owners and can consume the shared contract.
