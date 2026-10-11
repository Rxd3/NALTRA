# NALTRA

**Natural Language Analysis & Taxonomy Robust Architecture** studies multilingual
hierarchical multi-label classification of EU research-project objectives into EuroSciVoc
under translated (EN -> TR), code-switched and noisy input, with near-OOD rejection,
calibration and latency.

Results: [interactive results page](docs/results/index.html) (one self-contained HTML file;
download and open it, or serve `docs/` with GitHub Pages as in
[runbook section 6](docs/runbook.md#6-interactive-results-page)). How the numbers are
produced: [experiment runbook](docs/runbook.md).

## What is new

- **Data: the NALTRA CORDIS EN-TR benchmark, release 1.1.0** (CC BY 4.0,
  [data/LICENSE](data/LICENSE)). All 31,407 labelled CORDIS Horizon 2020 project objectives
  in English and in machine-translated Turkish (283,002 aligned sentence pairs), two
  synthetic EN-TR code-switch tracks (sentence- and chunk-level; validation and test) and a
  seeded noisy EN/TR test track (generated locally, not shipped). All tracks share one
  project-grouped, duplicate-aware split (21,985 / 4,711 / 4,711) and the same labels, and
  every shipped split is locked by the SHA-256 in its manifest ([Data](#data)).
- **Task: hierarchical multi-label classification under input shift, with rejection**
  ([Task](#task)): 473 direct labels in a 586-node tree, the same projects in four input
  conditions, near-OOD rejection, and course baselines vs new methods (sparse+dense RRF kNN,
  zero-shot Kev/Jev/Laya, NALTRA votes). Unlike flat, single-label categorisation, 84.3% of
  projects carry several labels and 59.8% span several roots.
- **Why Turkish.** It is agglutinative and morphologically rich (one stem takes many suffixed
  word forms), a hard case for word-level features; and we are not aware of any
  EuroSciVoc-labelled Turkish corpus.
- **Closest prior work.** EURLEX57K (Chalkidis et al. 2019: 57k English EU laws, about 4.3k
  EuroVoc labels) and MultiEURLEX (Chalkidis et al. 2021: 65k EU laws officially translated
  into 23 EU languages, EuroVoc labels, zero-shot cross-lingual transfer) are the closest
  resources we know of. NALTRA does not replace them; it differs in scope: research-project
  objectives labelled with the EuroSciVoc science taxonomy; Turkish, not an EU language,
  machine-translated rather than officially translated; synthetic code-switch and noisy
  copies paired with the same projects; near-OOD rejection; and zero-shot decision models
  against trained ones. Its trained members see Turkish in training, so EN -> TR is not
  zero-shot cross-lingual transfer.

Research questions:

1. *Which trained model fits 473 hierarchical labels best, and do votes help?* A calibrated
   linear SVM: 0.558 EN / 0.489 TR micro-F1, 0.668 hierarchical F1; no vote beats it (best:
   weighted soft vote, 0.523 / 0.472) ([Results](#results)).
2. *How much is lost on translated, code-switched and noisy input?* SVM loses 0.070 (-> TR),
   0.050 (-> sentence-level code-switch) and 0.078 (-> noisy EN), the multilingual hybrid kNN
   0.011, 0.009 and 0.019; SVM's drop is larger than every other member's in each tested
   condition (Holm p 0.0008), though Naive Bayes and XLM-R drop less than kNN on TR (point
   estimates, not tested) ([Findings](#findings)).
3. *Can confidence identify humanities-only projects after training excludes projects
   carrying retained humanities labels?* A retrained SVM reaches AUROC 0.927 EN / 0.924 TR
   (94 OOD vs 4,351 in-domain projects). Sub-threshold humanities codes remain in 165
   training, 19 validation and 34 nominally in-domain test projects, so this holds out
   retained labels, not the whole field ([Findings](#findings)).
4. *Do zero-shot decision models match trained ones?* On the 120-pair sample Jev cannot be
   separated from XLM-R, BiLSTM or Naive Bayes, nor Kev from XLM-R, all far below SVM
   (0.556); Laya scores 0.068 EN, a negative result. Observed pairwise micro-F1 intervals have
   average half-widths of about 0.043 EN and 0.041 TR; small differences remain uncertain
   ([Zero-shot sample](#zero-shot-sample)).

## Task

Given the objective text of an EU research project, predict every EuroSciVoc field it
belongs to. The task requires:

- **Ancestor-consistent multi-label output.** Each project gets a set of the 473 directly
  predicted labels in a 586-node, 6-root tree up to 7 levels deep: 3.2 direct labels on
  average (10.2 with ancestors); 84.3% of projects have more than one, and 59.8% (18,794 of
  31,407) span more than one root. Hierarchical F1 scores the ancestor-closed sets; the
  hierarchy-closed vote is ancestor-consistent by construction (members closed before voting).
- **Long-tail labels.** Median label support is 126 projects and 176 labels have fewer than
  100, so macro-F1 is reported next to micro-F1.
- **Paired language and input shift.** The same test projects are scored in English,
  machine-translated Turkish, synthetic EN/TR code-switched text and noisy copies; each drop
  is a paired bootstrap difference. Locally trained members see EN+TR training text, so
  EN -> TR measures the gap on translated input.
- **Near-OOD rejection.** Training excludes every project carrying a retained humanities
  label, and a model must flag the humanities-only test projects by low confidence (AUROC,
  FPR@95%TPR). Sub-threshold humanities codes remain in 165 training, 19 validation and 34
  nominally in-domain test projects, so this holds out retained labels, not the whole field.
- **Calibration.** Whether the scores can be read as probabilities (ECE, Brier).
- **Zero-shot vs trained.** Jev, Kev (one yes/no question per label) and Laya (one
  two-option question per label) are not trained or fine-tuned on CORDIS by the team
  (provider pretraining data unknown) and face every trained member on 120 test pairs.

## Data

Derived from one public source: **CORDIS - EU research projects under Horizon 2020
(2014-2020)**, CC BY 4.0, <https://cordis.europa.eu/data/cordis-h2020projects-csv.zip>. Of
35,389 project records, 31,407 have an objective and at least one supported label; they are
split by project into **21,985 / 4,711 / 4,711** train / validation / test, identical in
every track.

| Track | Location | Splits | Derivation |
| --- | --- | --- | --- |
| EN | `data/processed/cordis_h2020/en/` | train, validation, test | cleaned CORDIS objectives |
| TR | `data/processed/cordis_h2020/tr/` | train, validation, test | EN machine-translated by `Helsinki-NLP/opus-mt-tc-big-en-tr` @ `e539fc16` (pinned in `configs/data.yaml`); release 1.1.0 |
| Code-switch | `data/processed/cordis_h2020/code_switch/{sentence_mix,chunk_mix}/balanced/` | validation, test | aligned EN/TR sentences of the same project: alternating (`sentence_mix`) or first half / second half (`chunk_mix`) |
| Noisy | `data/noisy/combined/medium/cordis_h2020/{en,tr}/` (generated by `prepare_data.py --dataset noisy`, ignored) | test | one primary edit per document, drawn at random: adjacent-letter swap in 10% of words (35%), or duplication (35%) / deletion (30%) of 10% of letters and digits; plus occasional casing and punctuation changes (`combined`, `medium` = 0.10, seed 42; `src/naltra/data/noise.py` `_perturb_combined`) |

**Committed:** `data/raw/cordis_h2020/cordis-h2020projects-csv.zip` (the unmodified,
checksum-locked 55.2 MB archive), `data/processed/cordis_h2020/**/{*.jsonl.gz,manifest.json}`
(the cleaned splits, 100.9 MB compressed) and `data/splits/cordis_h2020/project_splits.json`
(the project split map). Restore the `.jsonl` files with:

```bash
python scripts/unpack_data.py      # or: make data
```

Each split must match the SHA-256 and size in its `manifest.json` before it replaces the
`.jsonl`; any mismatch fails without leaving a partial file. Details:
[dataset card](data/README.md) (intended use, known issues) and the full
[dataset specification](docs/dataset.md) (statistics in
[section 4](docs/dataset.md#4-multi-label--hierarchy-semantics)). TR and both code-switch
tracks are release 1.1.0: the 1.0.0 import with the 598 records its audit flagged
re-translated in 128-token chunks by the same checkpoint, labels unchanged. Every member is evaluated on 1.1.0 and every locally
trained one trained on it (`cordis_v0.5.0` `bilstm` and `transformer` refit their heads on
encoders from run v0.4.0, whose XLM-R fine-tuning data is not recorded).

## Preprocessing pipeline

| Step | What happens | Code (`src/naltra/data/`) |
| --- | --- | --- |
| 1. Archive check and extract | Reuses the committed archive (downloads only if absent), checks its SHA-256 against `configs/data.yaml`, extracts and checks `project.csv` and `euroSciVoc.csv` | `preprocessing.py` `download_cordis_h2020` |
| 2. Parse | Reads EuroSciVoc assignments; repairs project rows whose objective contains unescaped `;`, drops malformed rows | `preprocessing.py` `process_cordis_h2020` |
| 3. Clean objectives | Strips HTML tags with the pattern `<[^>]+>`, unescapes entities, collapses whitespace, keeps casing, drops empty objectives; the pattern also deletes `<…>` comparison spans that are not tags, in 62 retained projects ([details](docs/dataset.md#known-limitation-angle-bracket-comparisons), [Limitations](#limitations)) | `preprocessing.py` `preprocess_record_text` |
| 4. Label-support filter | Keeps codes marked direct-supported (>= 50 projects) in `taxonomy/taxonomy.json`, maps them through `taxonomy/label_map.json`, drops projects left without a label; sub-threshold codes are dropped, not rolled up to a supported ancestor ([Limitations](#limitations)); `scripts/build_taxonomy.py` rebuilds both files from the archive | `preprocessing.py` |
| 5. Hierarchy closure | `labels` = `labels_direct` plus every ancestor | `preprocessing.py` `hierarchy_closure` |
| 6. Duplicate-text grouping | SHA-256 fingerprint of the normalised, lowercased text; identical objectives form one group | `preprocessing.py` `compute_content_fingerprint` |
| 7. Stratified split | Multi-label stratification over fingerprint groups, so each project and each duplicate text sits in one split; 70/15/15, seed 42; writes the splits, `data/splits/cordis_h2020/project_splits.json` and a manifest | `splits.py`, `manifest.py` |
| 8. Translation | Sentence-aligned EN -> TR on the GPU with a resumable cache (`data/cache/translations/`); QA rejects empty, untranslated, control-character or length-ratio outliers; ids, split and labels are copied from EN | `translation.py` `translate_cordis_dataset` |
| 9. Code-switch and noise | Deterministic per-project mixing of aligned sentences; seeded character, casing and punctuation perturbation | `code_switching.py`, `noise.py` |
| 10. Audit | Manifest and source hashes, identity, fingerprints, project/pair/text leakage, full label coverage in train, EN/TR alignment | `validation.py` `validate_cordis_release` |

Rebuild from the committed archive (no download needed; translation took about 4 hours on
an RTX 3060):

```bash
python scripts/build_taxonomy.py                                    # taxonomy/*.json from the archive
python scripts/prepare_data.py --dataset cordis_h2020 --download   # steps 1-7
python scripts/validate_datasets.py --languages en
python scripts/prepare_data.py --dataset translation --device cuda  # step 8
python scripts/validate_datasets.py --languages en tr
python scripts/prepare_data.py --dataset noisy                      # step 9: EN/TR noisy test copies
python scripts/prepare_data.py --dataset code_switch
python scripts/validate_datasets.py --derived                       # step 10
```

**Shipped release.** Its 1.1.0 manifests lock the data-code hashes of the generating commit
(`ea00e6b`); that code has changed since, so plain `validate_datasets.py` refuses it and
points to `--prebuilt-release`: a record-level audit (identity, hierarchy closure,
fingerprints, leakage, EN/TR alignment, split SHA-256s) that finds 0 alignment mismatches and
0 QA failures (reported, not enforced) and reports segmenter drift; training and noisy
generation run it too. A rebuild rewrites the tracked manifests, split map and review sample
(train it without `--prebuilt-release`); TR 1.1.0 cannot be rebuilt, as its targeted repair
script is not in this repository ([details](docs/dataset.md#shipped-turkish-release-110)).

## Models

| Member | Role | Architecture | Model | Status |
| --- | --- | --- | --- | --- |
| `naive_bayes` | baseline | multinomial Naive Bayes over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k), one MultinomialNB per label, alpha 0.03 | Trained and scored; reported row, does not vote |
| `svm` | baseline | linear SVM over TF-IDF, one-vs-rest | same TF-IDF, one LinearSVC (C 1.0) per label, 3-fold project-grouped Platt calibration | Trained and scored |
| `bilstm` | baseline | bidirectional LSTM | 1 layer, 300-d embeddings and 50k vocabulary learned from training only, 256 hidden per direction; early stopping ended training after epoch 9 and kept the epoch-7 weights (lowest validation loss), then head refit on the frozen encoder; 512 tokens | Trained and scored (`cordis_v0.5.0`) |
| `transformer` | baseline | Transformer encoder (RoBERTa family) | `xlm-roberta-base` encoder fine-tuned 3 epochs on 256-token inputs (run v0.4.0), then head refit on the frozen encoder; 512 tokens | Trained and scored (`cordis_v0.5.0`) |
| `hybrid_knn` | new | sparse + dense retrieval kNN with reciprocal rank fusion (Cormack et al. 2009, k = 60) | TF-IDF cosine and `intfloat/multilingual-e5-base` neighbours (50 each), RRF k 60, then a label vote | Trained and scored (`cordis_v1.1.0_knn`) |
| `kev` | new | zero-shot yes/no decision model (open Jev alternative) | Kev-0.8B (`jaredpalmer/kev-0.8b` @ `bf75a6a8`), one yes/no question per label to a local server | Scored on 80 validation and 120 test project pairs (about 75 s per project on a GTX 1650); needs its server ([runbook](docs/runbook.md#5-kev-server-wsl2-ubuntu)) |
| `jev` | new | hosted System One decision model, zero-shot yes/no per label | TypeSafe `jev-latest`, the same 473 questions as Kev | Scored on the same project-pair samples as Kev; needs `JEV_API_KEY` and `JEV_API_BASE_URL=https://api.typesafe.ai` in `.env` |
| `laya` | new | local pretrained multilingual choice-question model, zero-shot | `convaiinnovations/laya` multilingual checkpoint @ `7b928d82` (laya SDK 0.4.1), one two-option question per label (A = the label's field name, B = a fixed "other" option), score P(A) | Scored on the project-pair samples as a non-voting member (0.068 EN / 0.046 TR micro-F1), from an artifact built by `scripts/prepare_laya.py` (about 145 s (EN) to 200 s (TR) per document on a GTX 1650, about 19 h for the 400 sample documents) |
| votes | new | NALTRA hard and soft votes | strict majority, tuned k-of-M, hierarchy-closed k-of-M; mean or val-A micro-F1-weighted mean of member scores; leave-one-out majority per member | Computed offline from the dumps, so no model is re-run to try another vote |

Every model family returns the same prediction contract, so votes are computed offline from
saved score matrices. In `cordis_v0.5.0` (teammate-trained) the fully trained `bilstm` and
`transformer` encoders are frozen and their heads refit on EN+TR with 512-token inputs. Every
locally trained model is fit on the 473 `labels_direct` targets of the EN+TR training data
(track `en_tr_direct`). Roles, architectures and models are fixed in `configs/benchmark.yaml`
and explained in [docs/evaluation.md](docs/evaluation.md#baselines-and-new-methods). "Laya"
in a `configs/models/hybrid_knn.yaml` comment refers only to the RRF constant k = 60.

## Evaluation

| Area | Measures |
| --- | --- |
| Classification | Micro/macro P, R, F1 over the full 473-label universe; Jaccard |
| Hierarchy | Hierarchical F1 over ancestor-expanded labels; violation rate of the closed vote |
| Ranking | P@1/3/5, R-precision, nDCG@5 |
| Calibration | Pooled ECE and Brier score over every document-label cell, top-1 ECE |
| Language shift | EN -> TR drop, code-switch (`sentence_mix`, `chunk_mix`) and noisy EN/TR sets |
| Near-OOD | Humanities root held out of training (`scripts/run_ood.py`, `naive_bayes` and `svm` retrained); test projects mixing humanities with other roots excluded; AUROC and FPR@95%TPR of `1 - max probability` |
| Efficiency | Warmed-up p50/p95/p99 latency, batched throughput, artifact size |

Validation is split by `pair_id`: one half tunes member thresholds, the other the ensemble
settings; test sets are only scored. Differences use a paired project-grouped bootstrap
(10,000 resamples, 95% percentile intervals; two-sided sign p-value with add-one correction,
floor 2/10,001). Holm families: each vote vs the best member (4 per test set), each
system's drops from English (5 on the full sets; 1 on the sample, so Holm changes nothing) and
SVM's drop minus each other member's (4 per condition). Zero-shot vs trained members are
tested on the sample ([below](#zero-shot-sample)); other member-vs-member comparisons,
including drop differences outside the SVM-minus-member family (two members other than SVM,
noisy TR, votes), are point estimates.

## Results

Release 1.1.0, test sets of 4,711 projects each; thresholds tuned on validation only;
micro-F1 is the primary metric because predicting no label at all already scores
99.3% per-label accuracy ([why](docs/evaluation.md#benchmark-and-metric-choice)). The full-set
and 120-pair sample numbers with intervals are on the [results page](docs/results/index.html).
The table shows the two best of the four votes on test; the hard votes `hard_k` and
`hard_majority` reach 0.438 and 0.370 on EN (`results/metrics/cordis_v1.1.0_ensemble/summary.md`).

| System | Role | EN | TR | Code-switch (sentence) | EN noisy | Macro-F1 EN | Hier. F1 EN | p50 CPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `svm` | baseline | **0.558** | **0.489** | **0.508** | **0.480** | **0.489** | **0.668** | 2.0 ms |
| weighted soft vote | new | 0.523 | 0.472 | 0.503 | 0.459 | 0.443 | 0.641 | - |
| soft vote | new | 0.497 | 0.455 | 0.482 | 0.445 | 0.422 | 0.628 | - |
| `hybrid_knn` | new | 0.393 | 0.382 | 0.383 | 0.374 | 0.312 | 0.545 | 545 ms |
| `bilstm` | baseline | 0.289 | 0.271 | 0.284 | 0.241 | 0.161 | 0.461 | 13.4 ms |
| `naive_bayes` | baseline | 0.278 | 0.275 | 0.280 | 0.262 | 0.198 | 0.440 | 1.8 ms |
| `transformer` | baseline | 0.248 | 0.240 | 0.244 | 0.203 | 0.119 | 0.438 | 376 ms |

p50 CPU is single-text latency over 200 EN test texts after 10 warm-up calls on an Intel
Core i5-9300H (CUDA hidden, 4 torch threads). Sizes and parameter counts in
`benchmark_summary.md` are not comparable across families (stored values: all three SVM
calibration folds, kNN index entries, without the 1.1 GB e5 encoder of `hybrid_knn`).

Micro-F1 lost from English test (paired bootstrap, 95% interval, Holm):

| System | -> TR | -> code-switch (sentence) | -> EN noisy | -> TR noisy |
| --- | --- | --- | --- | --- |
| `svm` | 0.070 [0.064, 0.075] | 0.050 [0.046, 0.055] | 0.078 [0.072, 0.084] | 0.126 [0.119, 0.133] |
| `hybrid_knn` | 0.011 [0.007, 0.016] | 0.009 [0.005, 0.014] | 0.019 [0.015, 0.023] | 0.026 [0.021, 0.031] |
| `transformer` | 0.008 [0.003, 0.013] | 0.003 [-0.001, 0.007], not significant | 0.044 [0.039, 0.049] | 0.053 [0.047, 0.059] |
| `bilstm` | 0.018 [0.012, 0.024] | 0.005 [0.001, 0.010] | 0.048 [0.042, 0.054] | 0.070 [0.063, 0.077] |
| `naive_bayes` | 0.002 [-0.002, 0.007], not significant | -0.002 [-0.005, 0.000], not significant | 0.016 [0.011, 0.021] | 0.027 [0.021, 0.034] |

SVM's drop minus each other member's drop is above 0 on TR, both code-switch tracks and noisy
EN: 16 differences from 0.030 [0.023, 0.038] (BiLSTM, noisy EN) to 0.067 [0.061, 0.074]
(Naive Bayes, TR), each with Holm p 0.0008, the smallest that four tests over 10,000 resamples
can give (`results/metrics/cordis_v1.1.0_ensemble/robustness.json`; noisy TR is not in this
test).

### Zero-shot sample

Zero-shot decision models on the fixed pair-hash sample (120 test project pairs, the same
projects for every system; thresholds and weights from 32 val-A pairs, ensemble settings
from 48 val-B pairs of an 80-pair validation sample). Six members vote here (Naive Bayes and
Laya are scored, never vote), so this weighted soft vote is not the full-set one, and
trained members' thresholds are re-tuned (SVM 0.12 vs 0.16 on the full sets). Kev, Jev and
Laya ask English questions (Kev and Jev: "Is this project about {name}?") of Turkish text
too, so their EN -> TR drop mixes input shift with cross-lingual prompting. 249 of the 473
direct labels have no positive project in the test sample and 304 none in the validation
sample, so sample macro-F1 (at most 224/473 = 0.474) is not comparable to the full test sets:

| System | Role | EN micro-F1 | TR micro-F1 | EN -> TR drop |
| --- | --- | --- | --- | --- |
| `svm` | baseline | 0.556 | 0.489 | 0.066 [0.033, 0.100] |
| weighted soft vote | new | 0.510 | 0.468 | 0.042 [0.016, 0.069] |
| `hybrid_knn` | new | 0.377 | 0.382 | -0.005 [-0.027, 0.018], not significant |
| `bilstm` | baseline | 0.298 | 0.278 | 0.020 [-0.020, 0.060], not significant |
| `naive_bayes` | baseline | 0.297 | 0.298 | -0.001 [-0.029, 0.026], not significant |
| `jev` | new | 0.277 | 0.279 | -0.002 [-0.018, 0.014], not significant |
| `kev` | new | 0.265 | 0.220 | 0.045 [0.021, 0.070] |
| `transformer` | baseline | 0.265 | 0.241 | 0.024 [-0.010, 0.057], not significant |
| `laya` | new | 0.068 | 0.046 | 0.021 [-0.005, 0.048], not significant |

### Findings

- A calibrated linear SVM is the most accurate member on every set, and none of the tested
  votes beats it: the best ensemble (weighted soft) is 0.036 [0.031, 0.040] micro-F1 below
  it on English.
- SVM, the most accurate member, also loses the most micro-F1: its drop exceeds each other
  member's on TR, both code-switch tracks and noisy EN (tested, Holm p 0.0008 each). On
  Turkish it drops 0.070 and the multilingual retrieval member (`hybrid_knn`) 0.011, a
  difference of 0.059 [0.052, 0.065]. `hybrid_knn` does not drop least: on Turkish, Naive
  Bayes (0.002 [-0.002, 0.007], not significant) and XLM-R (0.008 [0.003, 0.013]) drop less,
  as do BiLSTM, XLM-R and Naive Bayes on both code-switch tracks and Naive Bayes on noisy
  English, all from much lower English scores; these differences are not tested. Relative to
  its English score, SVM's noisy-English drop (14.0%) is smaller than BiLSTM's (16.6%) and
  XLM-R's (17.9%).
- Jev (not fine-tuned on CORDIS by the team; whether the provider's training data included
  CORDIS is unknown) cannot be separated on the English sample from XLM-R (+0.012 [-0.033,
  +0.058]), Naive Bayes (-0.020 [-0.062, +0.022]) or BiLSTM (-0.021 [-0.066, +0.025]). No
  statistically detectable micro-F1 difference was found between Kev and XLM-R on this sample
  (+0.000 [-0.046, +0.048]); on Turkish, Jev - XLM-R is +0.037 [-0.006, +0.080]
  (`results/metrics/cordis_v1.1.0_sample/pairwise.json`); these results do not establish
  equivalence. Jev shows no detectable EN -> TR drop (Holm p 0.85), which does not prove
  invariance; Kev drops 0.045 on Turkish.
- Laya, used without CORDIS fine-tuning, scores only 0.068 English / 0.046 Turkish on the
  sample (non-voting; its threshold 0.98, tuned on 32 val-A pairs, is the top of the
  0.02-0.98 grid in `src/naltra/evaluation/offline.py`): a negative result for this
  zero-shot setup. Fine-tuning was not evaluated.
- The neural members are weak (macro-F1 0.16 BiLSTM, 0.12 XLM-R), and XLM-R scores below
  Naive Bayes on English (0.248 vs 0.278 micro-F1): a training-budget result, not an
  architecture ranking (see [Limitations](#limitations)); the scores match the teammate's
  own validation selection, and no longer run was tried.
- Near-OOD (held-out humanities, 4,351 in-domain vs 94 OOD test projects): SVM AUROC 0.927
  [0.905, 0.946] EN and 0.924 [0.902, 0.945] TR; Naive Bayes 0.749 and 0.782.
- The prebuilt audit flagged 598 translation projects in the former TR 1.0.0 release; 1.1.0
  repairs them, and the audit flags 0 (`results/metrics/translation_exceptions.json`).

## Error analysis

Full test sets, tuned thresholds, point estimates without intervals
([results/metrics/cordis_v1.1.0_ensemble/error_analysis.json](results/metrics/cordis_v1.1.0_ensemble/error_analysis.json)):

- **Rare labels.** SVM's English micro-F1 grows with a label's EN-train support: 0.441 on tail
  labels (< 70 projects), 0.530 on mid (70-199) and 0.597 on head labels (>= 200).
- **XLM-R on rare labels.** 0.107 / 0.158 / 0.303 in the same buckets; only on head labels
  does it reach Naive Bayes (0.303 vs 0.299).
- **Single-label projects are hardest.** SVM English micro-F1 is 0.429 on the 730 projects
  with one gold label, against 0.564, 0.596 and 0.563 with 2, 3 and 4+.
- **Turkish hurts social sciences most.** SVM's EN -> TR drop is largest on social-sciences
  labels (0.529 -> 0.412); medical and health sciences is the best root (0.619 -> 0.541).
- **Over-predicted labels.** SVM's most frequent English false positive is "business
  models": 216 of its 375 predictions of that label are wrong.

## Limitations

- **Labels and text.** CORDIS labels are silver labels (CORDIS does not document how they
  are assigned); code-switched and noisy sets are synthetic. No human has rated the Turkish
  MT. Its evidence is structural QA on every record (non-empty, not a copy of the English,
  TR/EN length ratio 0.35-3.0, no control characters, aligned sentences; 0 failures on 1.1.0)
  and an AI-assisted rating of 100 sampled projects (meaning preserved: 83 yes, 17 partial, 0
  no; reviewer model and prompt not recorded; 18 rows show Turkish that no release ships)
  ([dataset specification section 8](docs/dataset.md#8-translation-pipeline--turkish-derivation)).
- **MT truncation and repetition.** Automatic screens flag 180 of the 31,407 TR records
  (0.57%; 31 in test) with a possibly truncated sentence (an aligned English sentence over 80
  characters whose Turkish keeps under 40% of its characters) and 48 (0.15%; 6 in test) with a
  possible repetition (a word n-gram, n >= 4, immediately repeated, the doubled sequence absent
  from the English); they are candidates, not confirmed errors, and their effect on scores is
  not measured (`results/metrics/translation_checks.json`,
  [dataset card](data/README.md#known-issues), item 7).
- **Label roll-up.** `labels_direct` keeps only codes with >= 50 supporting projects;
  sub-threshold codes are dropped, not rolled up to their nearest supported ancestor. 8,128
  of 31,407 retained projects lose at least one directly assigned code (9,909 codes). Within
  the active 586-node taxonomy, 1,039 of 4,711 test projects lack 2,375 CORDIS-implied nodes
  (327 miss a whole root). Restoring those active nodes changes hierarchical F1 by at most
  0.007 (SVM 0.668 -> 0.662) and no model ranking; full unpruned-tree performance was not
  evaluated. Counting the code paths the taxonomy pruned as well, 1,231 test projects lack
  3,744 (counted, not scored; `results/metrics/label_coverage.json`).
- **OOD scope.** One held-out root (humanities: 94 OOD vs 4,351 in-domain test projects),
  only `naive_bayes` and `svm` retrained, no far-OOD set; the 266 test projects with both
  humanities and other labels are excluded as ambiguous, so mixed-domain projects are untested.
- **Cleaning loss.** The step-3 tag pattern also deletes `<…>` spans that are comparisons,
  not HTML (`after (z<3) and before (z>3)` becomes `after (z 3)`): 62 retained projects (48
  train, 5 validation, 9 test) lose 17,290 characters. Labels are unaffected; dropping the 9
  test projects moves each member's full-set EN micro-F1 by at most 0.0002 (0.003 on the
  sample, which holds 2), and the effect on training is not measured. The 1.1.0 text is
  hash-locked, so a corrected cleaner needs a new release and a re-evaluation.
- **Single training run.** Every trained member is one run with seed 42, so the bootstrap
  intervals cover test-set sampling only, not training variance.
- **Label universe.** The >= 50-project support filter that defines the 473 labels counts
  all projects in the archive, test projects included (`scripts/build_taxonomy.py`), so the
  test split influenced which labels exist, though no model trains on it. Counted on the
  committed train and validation splits only, with the threshold scaled to their 85% share
  (>= 43 projects), 5 of the 473 labels would fall out (54 with an unscaled >= 50).
- **Validation reuse.** Member hyperparameters (NB alpha, neural checkpoints, head-refit
  candidates) were chosen on the full validation set, including the val-B half that tunes
  the ensembles, so ensemble settings may be slightly optimistic. Test sets were only scored.
- **Soft votes** average raw member scores on different scales (Platt probabilities,
  neighbour-vote fractions, sigmoid outputs, p(yes)); tuned thresholds are 0.12-0.16 for
  the voting trained members and 0.88 for Kev and Jev (the non-voting Naive Bayes: 0.22 on
  the full sets, 0.16 on the sample).
- **Neural training budget.** The XLM-R and BiLSTM rows measure training budgets, not the
  architectures' ceilings: XLM-R's encoder had 3 epochs on 256-token inputs with validation
  loss still falling (0.0382 to 0.0345), and BiLSTM learns its embeddings from scratch. The
  neural-vs-classical gap is therefore about budget, not architecture. The `xlm-roberta-base`
  revision was not recorded; the fine-tuned weights are in the artifact.
- **Sample-only members.** Kev, Jev and Laya run only on the 80 validation and 120 test pairs.
  Observed pairwise micro-F1 intervals have average half-widths of about 0.043 EN and 0.041
  TR; small differences remain uncertain. In English, Jev and Kev cannot be separated from
  XLM-R, BiLSTM or Naive Bayes, while on Turkish Kev is below Naive Bayes
  (-0.078 [-0.115, -0.041], Holm p 0.003) and BiLSTM (-0.058 [-0.099, -0.015], Holm p 0.045),
  and Laya is below every trained member in both languages
  (`results/metrics/cordis_v1.1.0_sample/pairwise.json`). Jev's `jev-latest` is a hosted
  alias whose weights can change, and the Kev dumps record the service configuration, not
  the served weights.
- **Near-duplicates.** Grouping catches only identical texts: 6, 15 and 25 of the 4,711 EN
  test projects have a training project at TF-IDF cosine of at least 0.95, 0.9 and 0.8
  (EN-train word 1-2-grams, min_df 2, sublinear tf; 7, 16 and 25 with the 50k vocabulary).

## Reproduce

Python 3.11 or newer (ranges in `pyproject.toml`). The tested versions (Python 3.13.1) are
pinned in [`requirements.txt`](requirements.txt) (the direct and dev dependencies plus the
runtime pins for SciPy, tokenizers, huggingface-hub, joblib, threadpoolctl and
safetensors), which needs Python 3.12 or newer. Use a git clone
(`git clone https://github.com/Rxd3/NALTRA`), not a ZIP: manifests and summaries record the
commit via `git rev-parse`. Commands are PowerShell (5.1 too: one per line, no `&&`), with
the bash form in a comment where it differs; `make` targets need Linux/macOS.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1                    # bash: source .venv/bin/activate
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu   # CPU-only: skips the 4 GB CUDA build
python -m pip install -r requirements.txt     # tested pins (the CPU torch above satisfies torch==2.14.1)
python -m pip install -e . --no-deps
# development install with version ranges instead: python -m pip install -e ".[dev]"
# GPU steps: python -m pip install --no-deps --force-reinstall torch==2.14.1+cu126 --index-url https://download.pytorch.org/whl/cu126
```

The NB/SVM equality below was checked with the tested pins.

**What is in Git.** The summaries behind every table are committed under `results/`:
`metrics/cordis_v1.1.0_ensemble/summary.{json,md}` (full sets),
`metrics/cordis_v1.1.0_sample/summary.{json,md}` (120-pair sample),
`ood/cordis_v1.1.0/summary.json`, `benchmarks/cordis_v1.1.0/benchmark_summary.{json,md}` and
`metrics/translation_exceptions.json` (`evaluate_ensemble.py` records the code commit,
environment and bootstrap seed in each `summary.json`), plus the `label_coverage.json`,
sample `pairwise.json`, `error_analysis.json`, `robustness.json` and
`translation_checks.json` of `scripts/extra_analyses.py`. Prediction
dumps and models stay local (too large). From a clone, the NB and SVM rows, their intervals
and the NB/SVM near-OOD results can be regenerated on CPU (about 1.5 GB of data, models and
results besides the virtual environment); every other table can be checked against the
committed summaries.

```powershell
$env:LOKY_MAX_CPU_COUNT = "4"                            # bash: export LOKY_MAX_CPU_COUNT=4
python scripts/unpack_data.py                            # verified splits (or: make data)
python scripts/validate_datasets.py --prebuilt-release   # record-level audit of the shipped release, about 2 min
python -m ruff check .                                   # lint (or: make lint)
python -m black --check .
# tests (no data, weights or GPU; about 20 min): one process per file, a failing file re-run once
Get-ChildItem tests\test_*.py | ForEach-Object { python -m pytest -q $_.FullName; if ($LASTEXITCODE -ne 0) { python -m pytest -q $_.FullName } }
# bash: for f in tests/test_*.py; do python -m pytest -q "$f" || python -m pytest -q "$f"; done
```

Why per file: on Windows a test file occasionally stops with exit code 139 / 0xC0000005 and
no traceback (a native crash at varying points, e.g. inside transformers or scikit-learn),
which is not a test failure; one process per file confines it, and the loop re-runs that file.
Elsewhere `python -m pytest` (or `make test`) runs the suite in one process.

```powershell
# NB and SVM on CPU, about 45 min on a busy laptop; train the two families in separate runs
python scripts/prepare_data.py --dataset noisy --prebuilt-release
python scripts/train_all.py --datasets cordis_h2020 --prebuilt-release --models naive_bayes --device cpu --output-dir models/cordis_v1.1.0   # about 4 min
python scripts/train_all.py --datasets cordis_h2020 --prebuilt-release --models svm --device cpu --output-dir models/cordis_v1.1.0           # about 7-13 min
# Windows: a training run can end with exit 139 / -1073741819 / 0xC0000005 and no traceback;
# if models/cordis_v1.1.0/cordis_h2020/en_tr_direct/<family>/naltra.json is missing, re-run that family's line
python scripts/predict_all.py --artifacts models/cordis_v1.1.0/cordis_h2020/en_tr_direct --models naive_bayes svm --sets en_validation tr_validation en_test tr_test cs_chunk_test cs_sentence_test en_test_noisy=data/noisy/combined/medium/cordis_h2020/en/test.jsonl tr_test_noisy=data/noisy/combined/medium/cordis_h2020/tr/test.jsonl --device cpu --output-dir results/predictions/cordis_v1.1.0
python scripts/evaluate_ensemble.py --predictions results/predictions/cordis_v1.1.0 --members naive_bayes svm --tune-sets en_validation tr_validation --test-sets en_test tr_test cs_chunk_test cs_sentence_test en_test_noisy=data/noisy/combined/medium/cordis_h2020/en/test.jsonl tr_test_noisy=data/noisy/combined/medium/cordis_h2020/tr/test.jsonl --output-dir results/metrics/cpu_check
python scripts/sample_sets.py --sets en_validation tr_validation --pairs 80 --output-dir data/samples/val80
python scripts/sample_sets.py --sets en_test tr_test --pairs 120 --output-dir data/samples/test120
python scripts/predict_all.py --artifacts models/cordis_v1.1.0/cordis_h2020/en_tr_direct --models naive_bayes svm --sets en_val_kev=data/samples/val80/en_validation.jsonl tr_val_kev=data/samples/val80/tr_validation.jsonl en_test_kev=data/samples/test120/en_test.jsonl tr_test_kev=data/samples/test120/tr_test.jsonl --device cpu --output-dir results/predictions/cordis_v1.1.0
python scripts/evaluate_ensemble.py --predictions results/predictions/cordis_v1.1.0 --members naive_bayes svm --tune-sets en_val_kev=data/samples/val80/en_validation.jsonl tr_val_kev=data/samples/val80/tr_validation.jsonl --test-sets en_test_kev=data/samples/test120/en_test.jsonl tr_test_kev=data/samples/test120/tr_test.jsonl --output-dir results/metrics/cpu_check_sample
python scripts/run_ood.py --models naive_bayes svm --languages en tr --output-dir results/ood/cpu_check
```

`evaluate_ensemble.py` needs two voting members, so this check lets NB vote; thresholds are
tuned per member and every bootstrap uses the same seed, so its NB and SVM rows and
intervals equal the committed full-set and sample ones. Which fields of
`results/*/cpu_check*` must match the committed summaries, and which differ by design, is
listed in [results/README.md](results/README.md#checking-the-cpu-re-run).

With every member's dumps ([runbook sections 2-4](docs/runbook.md#2-train)), rebuild the
summary, the results page and the extra analyses:

```bash
python scripts/evaluate_ensemble.py --predictions results/predictions/cordis_v1.1.0 --members svm hybrid_knn bilstm transformer --baselines naive_bayes --tune-sets en_validation tr_validation --test-sets en_test tr_test cs_chunk_test cs_sentence_test en_test_noisy=data/noisy/combined/medium/cordis_h2020/en/test.jsonl tr_test_noisy=data/noisy/combined/medium/cordis_h2020/tr/test.jsonl --output-dir results/metrics/cordis_v1.1.0_ensemble
python scripts/build_dashboard.py --summary results/metrics/cordis_v1.1.0_ensemble/summary.json --sample-summary results/metrics/cordis_v1.1.0_sample/summary.json --predictions results/predictions/cordis_v1.1.0 --test-sets en_test tr_test cs_chunk_test cs_sentence_test en_test_noisy=data/noisy/combined/medium/cordis_h2020/en/test.jsonl tr_test_noisy=data/noisy/combined/medium/cordis_h2020/tr/test.jsonl --latency results/benchmarks/cordis_v1.1.0/benchmark_summary.json --ood results/ood/cordis_v1.1.0/summary.json --output docs/results/index.html
python scripts/extra_analyses.py   # all five analyses; outputs in the Scripts table below
```

| Member or step | Needs beyond a CPU clone |
| --- | --- |
| `bilstm`, `transformer` | the teammate's `cordis_v0.5.0` weights, or GPU retraining with an `xlm-roberta-base` download |
| `hybrid_knn` | the `intfloat/multilingual-e5-base` download (1.1 GB) and, in practice, a GPU |
| `kev` | a WSL2 Kev server with the `jaredpalmer/kev-0.8b` weights ([runbook](docs/runbook.md#5-kev-server-wsl2-ubuntu)) and `KEV_API_KEY`, `KEV_API_BASE_URL` in `.env`; about 75 s per project on a 4 GB GTX 1650, hence the 120-pair sample |
| `jev` | `JEV_API_KEY` (paid hosted API) and `JEV_API_BASE_URL=https://api.typesafe.ai` in `.env` (copy `.env.example`; never commit `.env`) |
| `laya` | `pip install -r requirements-laya.txt`, the 678 MB pinned checkpoint (`prepare_laya.py`) and, in practice, a GPU: about 145 s (EN) to 200 s (TR) per document on the GTX 1650, about 19 h for the 400 sample documents (about 7 s per document on an RTX 4050), hence the sample ([runbook](docs/runbook.md#kev-jev-and-laya-on-fixed-samples)) |
| `run_benchmark.py` (latency) | the artifacts of all five trained families and an idle machine |

- Every `train_all.py` run rewrites `<output-dir>/training_summary.json`; check that it
  reports `"success"` before the next run overwrites it.
- After joblib's "A task has failed to un-serialize" (memory pressure with `n_jobs: -1`),
  re-run that family with `LOKY_MAX_CPU_COUNT=4`. After a native crash, a family whose
  `naltra.json` exists and whose status says `"success"` is usable (`predict_all.py` checks it).
- `train_all.py --smoke` trains a bounded subset offline for every family except
  `hybrid_knn`, which needs the e5 encoder in the Hugging Face cache.

## Repository map

| Path | Purpose |
| --- | --- |
| `contributions/` | One file per team member, `<NAME>_<ID>_<GITHUB>.md`: role, work done and results |
| `configs/` | Source and translation locks (`data.yaml`), training defaults (`training.yaml`), benchmark protocol (`benchmark.yaml`: primary metric, system roles), calibration bins of `evaluate_all.py` (`evaluation.yaml`); one YAML per member in `models/` |
| `data/raw/`, `data/processed/`, `data/splits/` | Committed CORDIS archive (extracted CSVs stay local); compressed EN, TR and code-switch splits with manifests; project split map |
| `data/review/` | Stratified 100-project translation review sample (its human-review columns still read `pending`) and its AI-assisted audit |
| `data/noisy/`, `data/cache/`, `data/samples/` | Generated noisy test copies, translation cache and the Kev, Jev and Laya samples (ignored) |
| `taxonomy/` | EuroSciVoc tree (`taxonomy.json`), source-code to label map (both rebuilt by `scripts/build_taxonomy.py`), and its validator |
| `src/naltra/data/` | Preparation: cleaning, splitting, translation, code-switch, noise, manifests, release audits, JSONL loading |
| `src/naltra/models/` | Shared `BaseNALTRAModel` contract, classical, neural and service bases, one package per member (`laya/` wraps the pinned local checkpoint) |
| `src/naltra/pipeline/` | Inference: normalisation, language detection, thresholds, hierarchy paths, OOD, ensemble voting |
| `src/naltra/evaluation/` | Metrics, hierarchical metrics, calibration, OOD, bootstrap and Holm, threshold tuning over dumps, latency, figure data, the resumable prediction cache of `evaluate_all.py` |
| `src/naltra/reporting/` | HTML template of the results page |
| `src/naltra/schemas/`, `src/naltra/utils/` | `PredictionResult` and related typed prediction structures; YAML config loading and run provenance (git commit, repository-relative paths) |
| `src/naltra/cli.py` | `prepare_data` entry point, shared by `scripts/prepare_data.py` and `naltra-prepare-data` |
| `scripts/` | Command-line entry points (next section) |
| `tests/` | Pytest suite on small fixtures (41 files: data and audits, models, evaluation and votes, scripts, results page) |
| `models/` | Trained artifacts, `<run>/<dataset>/<track>/<family>/` (ignored) |
| `results/` | Committed metric, OOD, benchmark, translation-exception and translation-check summaries (see [Reproduce](#reproduce)); prediction dumps, plots and OOD models are generated locally (ignored) |
| `docs/` | [Runbook](docs/runbook.md) (commands, hardware, Kev server), [dataset specification](docs/dataset.md) (the dataset card is [data/README.md](data/README.md)), [taxonomy](docs/taxonomy.md), [architecture](docs/architecture.md) (prediction contract, pipeline), [core ML](docs/core_ml.md) (training, artifacts, service adapters, local Laya), [evaluation](docs/evaluation.md) (metric choice, baselines vs new methods, code-switch evaluation), [team ownership](docs/team_ownership.md) |
| `docs/results/` | The built interactive results page (`index.html`) |

Top level: `pyproject.toml` (package, dependencies, pytest/Ruff/Black settings),
`requirements.txt`, `requirements-laya.txt` (optional pinned laya SDK), `Makefile`
(`install`, `data`, `test`, `lint`, `format`), `.env.example`, `LICENSE`,
[CONTRIBUTING.md](CONTRIBUTING.md) (pull-request workflow; area owners in
[docs/team_ownership.md](docs/team_ownership.md)).

## Scripts

All 17 scripts in `scripts/`, run from the repository root:

| Script | Inputs -> outputs |
| --- | --- |
| `unpack_data.py` | `data/processed/**/*.jsonl.gz` + manifests -> verified `.jsonl` splits |
| `build_taxonomy.py` | Raw archive -> `taxonomy/taxonomy.json` and `taxonomy/label_map.json` |
| `prepare_data.py` | Raw archive -> EN splits; `--dataset translation`, `code_switch` or `all` for derived tracks; `--dataset noisy` -> `data/noisy/combined/medium/cordis_h2020/{en,tr}/test.jsonl`; `--prebuilt-release` derives from the shipped 1.1.0 release |
| `validate_datasets.py` | Prepared release -> full audit pass or error; `--code-switch` adds both code-switch tracks, `--derived` noise and code-switch; `--prebuilt-release` audits the shipped 1.1.0 release record by record, which the full audit refuses ([why](#preprocessing-pipeline)) |
| `train_all.py` | Splits + `configs/` -> `models/<run>/cordis_h2020/<track>/<family>/`; rewrites `<run>/training_summary.json` on every run |
| `prepare_laya.py` | Pinned checkpoint in `configs/models/laya.yaml` -> `<output-dir>/laya/` artifact (checkpoint files with their SHA-256s, the 473 label questions); no training |
| `sample_sets.py` | Evaluation sets -> the `--pairs` projects with the smallest pair-id SHA-256 in each, `data/samples/<dir>/<set>.jsonl`, for Kev, Jev and Laya |
| `predict_all.py` | Artifacts + evaluation sets -> score matrices `results/predictions/<run>/<family>/<set>.npz` with `.json` sidecars |
| `classify.py` | Artifacts + `--text` or a JSONL `--input` -> one JSON line per text with labels, taxonomy-closed labels, language and OOD decision (`disabled` for the reported neural artifacts, whose head refit cleared the threshold); several `--models` are combined by a `--method` hard or soft vote, which uses a fixed 0.5 soft threshold (not the benchmark's tuned one) and reports no OOD decision |
| `evaluate_ensemble.py` | Score matrices -> tuned thresholds, member and ensemble metrics, bootstrap tests in `results/metrics/<run>/summary.{json,md}` |
| `evaluate_all.py` | The teammates' direct BiLSTM/XLM-R/Laya evaluator with `--resume`; it runs the full validator, so it needs a release rebuilt from the raw archive, and no reported number comes from it |
| `translation_exceptions.py` | Summary + dumps + a `--prebuilt-release` artifact -> the projects the prebuilt audit flagged (0 on 1.1.0) and scores with and without them, `results/metrics/translation_exceptions.json` |
| `extra_analyses.py` | Committed data + local prediction dumps (`coverage`, `pairwise`, `errors`, `drops`, `translation`; default all) -> label coverage against the CORDIS closure within the active 586-node taxonomy, plus unpruned code-path counts (`results/metrics/label_coverage.json`), sample member pairs (`results/metrics/cordis_v1.1.0_sample/pairwise.json`), `results/metrics/cordis_v1.1.0_ensemble/error_analysis.json`, each member's EN micro-F1 drop on TR, code-switch and noisy EN with SVM's drop minus each other member's (paired bootstrap, Holm; `results/metrics/cordis_v1.1.0_ensemble/robustness.json`) and, from the TR splits alone, the truncation and repetition screens (`results/metrics/translation_checks.json`) |
| `make_figures.py` | Summary + score matrices -> paper figures `results/plots/<run>/*.png` |
| `run_benchmark.py` | Artifacts -> latency, throughput and size in `results/benchmarks/<run>/benchmark_summary.*` |
| `run_ood.py` | Splits -> retrains NB/SVM without one root (default humanities), near-OOD AUROC and FPR@95%TPR in `<output-dir>/summary.json` (reported: `results/ood/cordis_v1.1.0`); refuses an existing `--output-dir` unless it holds an earlier `run_ood.py` output or `--overwrite` is given, and always refuses one that overlaps `--data-dir` or the repository root |
| `build_dashboard.py` | Summary + 120-pair sample summary + dumps + latency + OOD -> `docs/results/index.html` |

## References and licences

Code: MIT ([LICENSE](LICENSE)). Data and taxonomy: CC BY 4.0 ([data/LICENSE](data/LICENSE)),
derived from "European Union, CORDIS - https://cordis.europa.eu - CC BY 4.0". Third-party
models are downloaded at their pinned revisions, not redistributed:

| Resource | Used for | Licence |
| --- | --- | --- |
| CORDIS H2020 projects (European Union) and EuroSciVoc (Publications Office of the EU) | texts and labels | CC BY 4.0 ([data/LICENSE](data/LICENSE)) |
| `Helsinki-NLP/opus-mt-tc-big-en-tr` @ `e539fc16` (OPUS-MT; Tiedemann and Thottingal 2020) | EN -> TR translation | CC BY 4.0 |
| `intfloat/multilingual-e5-base` @ `d1287505` (Wang et al. 2024) | `hybrid_knn` dense retriever | MIT |
| `xlm-roberta-base` (XLM-R; Conneau et al. 2020) | `transformer` | MIT |
| `convaiinnovations/laya` @ `7b928d82`, laya SDK 0.4.1 | `laya` | Apache-2.0 |
| `jaredpalmer/kev-0.8b` @ `bf75a6a8` on `Qwen/Qwen3.5-0.8B-Base` | `kev` | Apache-2.0 |
| TypeSafe `jev-latest` (hosted API) | `jev` | the provider's terms of service |

- Chalkidis, I. et al. (2019). Large-scale multi-label text classification on EU legislation. ACL.
- Chalkidis, I., Fergadiotis, M. and Androutsopoulos, I. (2021). MultiEURLEX - A multi-lingual and multi-label legal document classification dataset for zero-shot cross-lingual transfer. EMNLP.
- Conneau, A. et al. (2020). Unsupervised cross-lingual representation learning at scale. ACL.
- Cormack, G. V., Clarke, C. L. A. and Buettcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. SIGIR.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6(2), 65-70.
- Tiedemann, J. and Thottingal, S. (2020). OPUS-MT: building open translation services for the world. EAMT.
- Wang, L. et al. (2024). Multilingual E5 text embeddings: a technical report. arXiv:2402.05672.
