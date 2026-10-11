# NALTRA codebase and results audit

Historical review of the producing checkout, with a follow-up on the critical fixes.
For current setup and artifact handoff instructions, see [documentation](README.md)
and [reproducibility](reproducibility.md).

Follow-up: the requested high-priority code fixes address findings **1, 2 and 4** below.
Prediction dumps now validate batch length, text order, model identity and complete finite
scores. Wheels include canonical configs/taxonomy resources, verified by installed-model
inference outside the checkout. `classify.py --benchmark-settings summary.json` imports
frozen thresholds and votes after artifact/taxonomy checks. The original audit below records
the state before these fixes; dataset files and published results have not been changed.
The 11 October integration review incorporates the team's subsequent Laya sample
result (0.068 EN / 0.046 TR micro-F1, non-voting); statements below about pending
Laya results describe the original 10 October checkout.
Integration verification: **844 tests passed, 1 expected failure**; Ruff, Black,
all ten split checksums, and the prebuilt EN/TR record audit passed. The Laya
summary preserves all earlier sample metrics and voting members. Documentation
links and regenerated chart-source fingerprints were also checked. Original
model artifacts and prediction dumps remain unavailable for independent scoring.
Follow-up verification: **835 tests passed, 1 expected failure**; Ruff, Black and
`git diff --check` passed. The expected failure remains the documented scientific-text
cleaning limitation, which requires a separate dataset release.

Reviewed 10 October 2026, checkout `000bc21`. This is a broad source, structure, test, data, packaging, and result-consistency review. No application code was changed. The missing runtime dependencies were installed in an isolated environment under the Windows temporary directory. The ten committed compressed splits were restored to their Git-ignored JSONL locations.

## Assessment

NALTRA is a well documented research codebase with substantial automated coverage and useful safeguards against data leakage. Its strongest implementation areas are the shared prediction contract, project-grouped data checks, calibrated classical models, reproducible input variants, and offline evaluation. Its largest weaknesses are incomplete experiment reproducibility, separation between benchmark settings and usable inference, and a few missing integrity checks. The evidence supports using it for continued research; packaging and inference need attention before treating it as a portable application.

Passing tests do not establish that the published model outputs are correct. The trained weights and prediction dumps are absent from this checkout, so I could verify data and published-summary consistency but could not independently recompute their classification scores or timings. I did not retrain the full models, download pretrained checkpoints, or call paid services.

## Findings, ordered by priority

### 1. High: prediction-dump generation can silently misalign or omit predictions

Location: `scripts/predict_all.py:93`.

`score_set()` allocates an uninitialized `np.empty` matrix, then trusts the number and order of results from `predict_batch()`. It does not verify batch length, returned text, model identity, or the exact returned label universe. A short batch leaves unwritten rows; a reversed batch attaches scores to the wrong IDs. The dump subsequently receives the intended input IDs, so the downstream ID and input-file checks cannot detect that the model actually scored different rows.

Confirmed with a stub that returned only the second document's prediction for a two-document batch: the function returned a two-row matrix successfully and placed that prediction in row one. This proves the missing guard, not that any shipped model produced such output.

Fix: validate exact batch length and pair each result with its normalized input using strict iteration; require the expected model and complete label set; reject nonfinite scores before publishing the dump. Publish the NPZ and sidecar through temporary files so failed writes do not leave a half-complete pair.

### 2. High: a built wheel installs but its models cannot initialize

Locations: `pyproject.toml:36`, `src/naltra/data/manifest.py:20`, `src/naltra/models/classical.py:71`, `src/naltra/pipeline/prediction.py:49`.

Runtime configuration and taxonomy paths are derived from source-file ancestry and assume a repository checkout. Setuptools discovers only the packages under `src`; the built wheel contains no YAML, JSON, or HTML resources.

Confirmed by building and installing `naltra-0.1.0-py3-none-any.whl`, then running `SVMModel()` outside the checkout. It fails with `FileNotFoundError` for `.../Lib/configs/models/svm.yaml`. The wheel build itself succeeds, which makes this easy to miss in an editable-install workflow.

Fix: ship immutable defaults and taxonomy as package resources accessed with `importlib.resources`, and configure writable dataset/artifact roots explicitly. Add a wheel-install smoke check outside the repository. If distribution is intentionally unsupported, make that limitation explicit and avoid presenting the installed entry point as portable.

### 3. High for research reproducibility: the full published experiment cannot be independently replayed from this checkout

Locations: `models/README.md:3`, `results/README.md:11`, `docs/dataset.md:326`, `README.md:64`.

The checkout contains model documentation but no trained artifacts or score matrices. The frozen Turkish 1.1.0 repair script is missing, so ordinary translation produces a different release. The README also states that the shipped XLM-R encoder's original fine-tuning data is not recorded. Refitting a classifier head does not recover the encoder's training lineage.

These limitations are disclosed; I found no evidence that the numbers were fabricated or that the encoder actually saw test projects. However, the available evidence cannot establish the complete neural training lineage or independently verify all published scores.

Fix: provide a versioned artifact bundle outside Git containing weights, tokenizers, prediction dumps, exact commands, resolved configurations, input hashes, original encoder training manifests, and the repair implementation. Provide immutable download locations and checksums. Otherwise, label the release as an archived experiment with partial reproducibility.

### 4. Medium: classification does not reproduce the benchmark's decisions

Location: `scripts/classify.py:116`.

The CLI retains each loaded artifact's thresholds and constructs ensembles from defaults. It has no option to import the offline benchmark settings. The published full-set thresholds are SVM 0.16, kNN/BiLSTM/Transformer 0.14, and NB 0.22; current training defaults use 0.5. The CLI's soft ensemble also uses 0.5, while the published soft and weighted-soft thresholds are 0.14 and 0.16. The CLI supports hard/soft methods but does not expose the reported weighted-soft or tuned k-of-M systems.

The documentation explains part of this difference, but someone trying the CLI should not expect the reported benchmark behavior.

Fix: export a validated inference configuration from the benchmark summary and allow the CLI to load its thresholds, weights, member identities, and taxonomy identity. Keep threshold provenance in prediction output.

### 5. Medium: scientific comparison expressions are removed during cleaning

Locations: `src/naltra/data/preprocessing.py:47`, `tests/test_preprocessing.py:63`.

The regular expression `<[^>]+>` treats mathematical comparisons as HTML. Confirmed: `Galaxies after (z<3) and before (z>3) are compared.` becomes `Galaxies after (z 3) are compared.` The repository documents 62 affected retained projects and marks the regression test as an expected failure.

Fix: use tag-aware cleaning with regression coverage for scientific notation, then issue a new dataset release and reevaluate. Preserve the current hash-locked release for historical comparisons.

### 6. Medium: neural artifact loading has weaker corruption detection than classical and Laya loading

Locations: `src/naltra/models/neural.py:568`, `src/naltra/models/neural.py:590`.

The shared neural payload binds the taxonomy but does not store and verify hashes of its weights and tokenizer/vocabulary files. Shape-compatible changed weights can therefore load without an integrity mismatch. Classical artifacts verify `estimator_sha256`, and Laya verifies checkpoint hashes. Downstream prediction provenance hashes the artifact as it exists at scoring time; it does not prove that it matches the originally saved neural artifact.

Fix: store a complete file-hash inventory in the neural metadata and validate it before loading. Hashes detect corruption; they do not make untrusted artifacts safe. The existing warning about trusted joblib artifacts is appropriate.

### 7. Low: the committed timing summary lacks provenance fields claimed by its documentation

Locations: `results/README.md:13`, `results/benchmarks/cordis_v1.1.0/benchmark_summary.json:1`, `scripts/run_benchmark.py:199`.

The README says the timing summary records `code_commit` and top-level `environment`. The committed file has neither. It does contain hardware, runtime package versions, native settings, test-file hash, and artifact hashes, so useful provenance remains. The current writer includes the missing fields, indicating that the committed artifact predates that change.

Fix: regenerate the timing summary or explicitly document its older provenance schema. Do not invent a historical producing commit.

## Verification performed

- Full pytest suite: **807 passed, 1 xfailed**, in 132.69 seconds. The expected failure is scientific comparison cleaning. Real headless dashboard tests executed without skips.
- Ruff: passed. Black: all 115 Python files unchanged.
- Wheel: built and installed successfully; outside-checkout model initialization failed as described above.
- All ten compressed dataset splits decompressed with matching manifest SHA-256 and size checks.
- EN/TR prebuilt audit: 21,985 train, 4,711 validation, 4,711 test per language; no detected project/pair/content leakage; hierarchy and direct-label support checks passed.
- Translation audit: 15,434 segmenter-drift records, zero alignment/text mismatches, zero automated QA failures. These automated checks do not establish semantic translation quality.
- Raw archive SHA-256 matched `configs/data.yaml`.
- Taxonomy: 586 nodes and 473 directly supported labels.
- Both code-switch strategies, in both validation and test, matched current generation record by record after alignment by ID. Generator input order differs from stored order, so this was an ID-aligned fidelity check.
- Regenerated EN/TR noisy test JSONL hashes matched the full result summary. Regenerated 80-pair validation and 120-pair test sample hashes matched the sample summary.
- The generated dashboard's headline metric payload exactly matched the full committed metric summary.

The audit environment used Python 3.13.14, PyTorch 2.14.1+cpu, Transformers 5.19.0, NumPy 2.5.3, and scikit-learn 1.9.1. It was not the exact historical CUDA environment; some transitive and developer-tool versions differed. All application tests passed in this environment.

## What the results support

| System | EN micro-F1 | TR micro-F1 | EN macro-F1 | EN hierarchical F1 |
| --- | ---: | ---: | ---: | ---: |
| SVM | 0.5584 | 0.4886 | 0.4893 | 0.6683 |
| Weighted soft vote | 0.5228 | 0.4723 | 0.4426 | 0.6408 |
| Hybrid kNN | 0.3929 | 0.3817 | 0.3121 | 0.5446 |
| BiLSTM | 0.2891 | 0.2708 | 0.1610 | 0.4606 |
| Transformer | 0.2476 | 0.2398 | 0.1194 | 0.4382 |
| Naive Bayes | 0.2777 | 0.2753 | 0.1976 | 0.4397 |

These are committed results, not fresh model evaluations.

SVM is the strongest tested member across all six full test conditions. The best tested ensemble is worse on English by 0.0355 micro-F1, with a reported paired 95% interval of [-0.0401, -0.0310]. The experiment does not demonstrate an ensemble accuracy advantage.

Hybrid kNN's smaller EN-to-TR drop is interesting, but its absolute performance remains considerably below SVM. The neural rows describe the particular budgeted training runs and head refits; they do not establish an architectural ceiling. Transformer is below Naive Bayes on full English test micro-F1.

SVM's humanities OOD AUROC is 0.9269 EN and 0.9242 TR. At 95% OOD recall, false-positive rates are 0.3174 and 0.2825: approximately 28–32% of in-domain projects would be rejected at that operating point. Strong AUROC does not establish a satisfactory practical rejection threshold. Only one root is held out, with 94 OOD projects per language and 266 mixed-root projects excluded.

Kev/Jev results use 120 paired test projects, not the full 4,711. The absent-label problem makes sample macro-F1 particularly difficult to compare with full-set macro-F1. Laya has an implementation but no committed benchmark result.

Very low pooled calibration errors should be interpreted alongside top-1 calibration and accuracy because almost all document-label cells are negative. For example, SVM's pooled ECE is 0.0015, but top-1 ECE is 0.1061. The reporting code appropriately includes both.

The documentation correctly discloses validation reuse, taxonomy support filtering over the complete archive, synthetic language/noise shifts, pending human translation review, and one training seed. Test-project bootstrap intervals quantify test sampling uncertainty, not variation across training runs.

## Structure and maintenance

The broad separation among `data`, `models`, `pipeline`, `schemas`, and `evaluation` is sensible. The common model contract and separate offline votes reduce coupling between expensive inference and experimentation. Testing is substantial: 41 test files and roughly 9,400 lines, compared with roughly 11,200 lines of Python implementation in `src` and `scripts`.

The main structural problem is reusable behavior living in script entry points. Evaluation and dashboard scripts import `scripts.predict_all`, which imports `scripts.train_all` and its neural registry. Even sample generation therefore pulls in the training/model stack. Move set resolution, model registry/loading, artifact identity, and offline benchmark orchestration into the package, leaving scripts as argument parsing and calls into those modules. This also makes a lighter CPU-only installation feasible; currently PyTorch and Transformers are mandatory dependencies.

Manifest validation hashes every data module plus both dependency files and demands equality with the current checkout. This is conservative but conflates verifying an immutable release with regenerating it using current code. An unrelated development dependency update can invalidate the strict release audit. Separate historical release verification from generator reproducibility, and retain explicit status for tolerated translation issues. The current prebuilt audit reports QA/alignment failures without enforcing them; its successful status should not be read as a clean translation-quality guarantee.

No committed CI workflow was found. Add automated lint/test checks, a wheel-install check, and a lightweight committed-release integrity job. The existing tests are valuable; CI would make them a reliable merge gate. Benchmark size reporting should distinguish index/weight entries from neural parameters and include externally cached encoder size where comparisons require it; the README already acknowledges the current limitation.

## Recommended sequence

1. Fix prediction batch validation and export benchmark settings for inference.
2. Correct wheel resources and add the outside-checkout installation check.
3. Archive the complete published experiment and recover the missing generation/training lineage.
4. Add neural artifact hashes and CI.
5. Build a new release with corrected scientific-text cleaning, then evaluate calibrated ensembles and multiple training seeds under a cleaner validation protocol.
