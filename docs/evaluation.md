# Evaluation

NALTRA evaluates every model using fixed project splits and complete scores over
the same 473 direct labels. English, Turkish, code-switch, and noisy variants carry
their source `pair_id`, enabling paired comparisons.

## Protocol

1. Score each model once and save an NPZ matrix plus JSON metadata.
2. Split clean EN/TR validation by project pair into val-A and val-B.
3. Tune each member's threshold on val-A for micro-F1.
4. Tune ensemble vote counts and soft thresholds on val-B. Weighted soft voting
   uses each member's val-A micro-F1 as its weight.
5. Apply those decisions to test predictions without further tuning.
6. Compute paired project bootstrap comparisons and write summaries.

The full benchmark uses SVM, hybrid kNN, BiLSTM, and Transformer as voters.
Naive Bayes is a non-voting baseline, avoiding a second vote from the same TF-IDF
representation as SVM. All five trained models see EN+TR training text, so EN -> TR
measures the gap on translated input, not cross-lingual transfer.

Kev, Jev, and Laya are scored only on fixed 80-validation/120-test project-pair
samples, together with the five trained models; see the
[sample protocol](#sample-protocol-kev-jev-laya).

Member hyperparameters, neural checkpoints, and head-refit candidates were selected
using the full validation set before this val-A/val-B protocol
([tuning budget](#tuning-budget)). This validation reuse is a limitation; test sets
were only scored.

## Metrics

| Metric | Interpretation |
| --- | --- |
| Micro-F1 — primary | Pools correct/missed/extra direct-label assignments across records |
| Macro-F1 | Gives each of the 473 direct labels equal weight; absent positives count as zero |
| Hierarchical micro-F1 | Expands targets and predictions to include ancestors |
| Jaccard | Overlap of the predicted and target label sets |
| P@1/3/5, R-precision, nDCG@5 | Ranking quality without a selection threshold |
| Pooled ECE and Brier | Calibration over all document-label cells |
| Top-1 ECE | Calibration of the highest-scored label |
| OOD AUROC and FPR@95%TPR | Held-out-root detection ranking and false alarms |
| p50/p95/p99 latency and throughput | Runtime under recorded hardware and batch settings |

Predicting no labels yields 99.32% per-label accuracy because positives are sparse,
so accuracy is not the primary comparison. The median label occurs in 19 of 4,711
test projects; 386 labels have prevalence below 1%.

Pooled ECE is dominated by near-zero negative cells. Read it alongside top-1 ECE
and classification quality. Soft votes combine scores on different scales without
a separate recalibration step.

The sample test has no positives for 249 labels, and sample validation has none for
304. Sample macro-F1 therefore cannot exceed 224/473 = 0.474 and should not be
compared directly to full-set macro-F1.

`hard_closed` votes over the 586-node ancestor-expanded universe, so only its
hierarchical metrics are comparable to ordinary member/ensemble results.

Hierarchical closures start from the 473 retained labels only: CORDIS codes with
fewer than 50 supporting projects are dropped, not rolled up to a supported ancestor
(counts and effect on hierarchical F1 in [known issue 2](../data/README.md#known-issues)).

## Uncertainty and significance

The benchmark uses 10,000 paired bootstrap resamples grouped by `pair_id`,
95% percentile intervals, and Holm correction within each comparison family.
Results are in each summary's `significance` block. The p-value is the two-sided
bootstrap sign proportion with an add-one correction,
`min(1, 2(1 + min(#d <= 0, #d >= 0)) / (B + 1))` over the B resampled differences d,
so it is never below 2/10,001 (`src/naltra/evaluation/bootstrap.py`).

Project pairing keeps language variants together in resampling. Intervals describe
test-set sampling uncertainty for the recorded training run; they do not measure
training variance. Sparse sample macro-F1 intervals need care because bootstrap
samples can omit rare labels while retaining the fixed denominator.

### What is significance-tested

Holm adjusts each family separately, per metric:

| Family | Comparisons | Where |
| --- | --- | --- |
| Votes vs best member, per test set | The 4 votes (`hard_majority`, `hard_k`, `soft`, `weighted_soft`) minus the member with the best val-A micro-F1 (`svm`) | `summary.json["significance"]["vs_best_member"]` |
| Each system across conditions | First test set minus each other test set: 5 on the full sets (TR, both code-switch, both noisy), 1 on the sample (TR), where Holm therefore changes nothing | `summary.json["significance"]["cross_condition"]` |
| Zero-shot vs trained members, per sample language | Kev, Jev and Laya each minus each of the 5 trained members (15 tests) | `results/metrics/cordis_v1.1.0_sample/pairwise.json` (`scripts/extra_analyses.py pairwise`) |
| SVM's drop vs each other member's, per condition (micro-F1 only) | SVM's EN-minus-condition drop minus that of `hybrid_knn`, `bilstm`, `transformer` and `naive_bayes` (4 tests) on `tr_test`, `cs_sentence_test`, `cs_chunk_test` and `en_test_noisy`; one shared resample of `pair_id` for both sets and all members; `tr_test_noisy` is not in this family | `results/metrics/cordis_v1.1.0_ensemble/robustness.json` (`scripts/extra_analyses.py drops`) |

The results of the SVM drop family are in
[language and noise robustness](results.md#language-and-noise-robustness). The per-member
drops in the same file are the `cross_condition` ones (the script stops if a point
estimate differs by more than 1e-6).

The near-OOD AUROCs carry bootstrap intervals (`results/ood/cordis_v1.1.0/summary.json`)
but no test between models. Not tested at all, so only descriptive: differences between
two members on the full sets (for example XLM-R vs Naive Bayes), differences between two
systems' drops outside the SVM-minus-member family (for example `hybrid_knn`'s EN -> TR
drop, 0.011, vs Naive Bayes's, 0.002, any drop difference on noisy TR, or a vote's drop
vs a member's), relative drops, `hard_closed`, leave-one-out, calibration and latency.

## OOD and efficiency

The near-OOD experiment removes projects touching the humanities root from both
training and validation, then retrains NB and SVM. Test projects with only
humanities labels are OOD; projects with no humanities label are in-distribution.
Mixed projects are excluded.

Per language, this gives 4,351 in-distribution, 94 OOD, and 266 excluded projects.
The split looks at retained labels only: 165 train and 19 validation projects with a
sub-threshold CORDIS humanities code stay in its training data, and 34 of the 4,351
in-distribution test projects carry one.
The score is `1 - max(label probability)`. AUROC uses a stratified 2,000-resample
bootstrap; FPR@95%TPR is the share of in-distribution projects flagged when 95% of
OOD projects are caught. This experiment covers one nearby domain shift, rather
than arbitrary far-OOD text.

Latency uses 200 English test texts after 10 warm-up calls on an Intel Core
i5-9300H, with CUDA hidden and four torch threads. Remeasure on an idle machine.
Stored parameter counts/index entries are not comparable across architectures;
hybrid kNN artifact size excludes its separately downloaded e5 encoder.

## Model configurations

`configs/benchmark.yaml` defines each system's role, architecture, model, and
primary/secondary metrics. The baseline/new labels describe the course comparison.

| system | role | architecture | model |
|---|---|---|---|
| `naive_bayes` | baseline | multinomial Naive Bayes over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k, min_df 2, sublinear tf), one MultinomialNB per label, alpha 0.03 |
| `svm` | baseline | linear SVM over TF-IDF, one-vs-rest | same TF-IDF, one LinearSVC (C 1.0) per label, 3-fold project-grouped Platt calibration |
| `bilstm` | baseline | bidirectional LSTM | release `cordis_v0.5.0`: 1 layer, 300-d embeddings learned from scratch (50k vocabulary), 256 hidden per direction, dropout 0.2, trained with early stopping (patience 2, at most 10 epochs): it stopped after epoch 9 and kept the epoch-7 checkpoint, which had the lowest validation loss; then its sigmoid head refit on the frozen encoder (unweighted BCE, lr 1e-3); max_length 512 |
| `transformer` | baseline | Transformer encoder (RoBERTa family) | release `cordis_v0.5.0`: `xlm-roberta-base` fine-tuned 3 epochs (lr 2e-5), then its sigmoid head refit on the frozen encoder for up to 20 epochs (unweighted BCE, lr 1e-3); max_length 512 |
| `hybrid_knn` | new | sparse + dense retrieval kNN with reciprocal rank fusion (Cormack et al. 2009, k = 60) | TF-IDF cosine + `intfloat/multilingual-e5-base` embeddings, 50 neighbours per retriever, RRF k 60 |
| `kev` | new | zero-shot yes/no decision model (open-source Jev alternative) | Kev-0.8B (`jaredpalmer/kev-0.8b` at revision `bf75a6a8848ea6960ff2ed108d9ed44c2941174f`, served via the tag `v1.0`), p(yes) for "Is this project about {name}?" per label |
| `jev` | new | hosted System One decision model (Jev), zero-shot yes/no per label | TypeSafe `jev-latest`, asked the same "Is this project about {name}?" questions as Kev |
| `laya` | new | local pretrained multilingual choice-question model (Laya), zero-shot | `convaiinnovations/laya` multilingual checkpoint at revision `7b928d828b7b0e022f929d9bd2e44165aa270148` via laya SDK 0.4.1; one two-option question per direct label (A = the topic, B = other research topics), P(A) as the label score |
| `hard_majority`, `hard_k` | new | NALTRA hard vote | strict majority; k-of-M with k tuned on val-B |
| `hard_closed` | new | NALTRA hierarchy-closed hard vote | k-of-M over ancestor-closed 586-node label sets |
| `soft`, `weighted_soft` | new | NALTRA soft vote | mean, or val-A micro-F1-weighted mean, of raw member scores (each on its own scale, not recalibrated); threshold tuned on val-B |
| `leave_one_out` | new | NALTRA hard vote (ablation) | strict majority of the members left after dropping each one |

`hybrid_knn` is unrelated to the `laya` member: the comment "Laya uses k=60" in
`configs/models/hybrid_knn.yaml` refers only to the RRF constant.

### Tuning budget

What each system searched, and on which data. "Validation" is the full EN+TR validation
set (both halves); thresholds come from the 49-value grid 0.02-0.98 (`THRESHOLD_GRID` in
`src/naltra/evaluation/offline.py`). Every trained member is one run with seed 42.

| System | Searched (selection data) | Fixed, not searched |
| --- | --- | --- |
| `naive_bayes` | alpha (validation sweep: 0.03 chosen, micro-F1 0.27 vs 0.12 at alpha 1.0); threshold (val-A) | TF-IDF settings |
| `svm` | threshold (val-A) | C 1.0, TF-IDF settings, 3 calibration folds |
| `bilstm` | checkpoint by validation loss (early stopping, at most 10 epochs, patience 2); weighted vs unweighted head refit (validation); threshold (val-A) | architecture, lr 1e-3, batch 32 |
| `transformer` | checkpoint by validation loss within 3 epochs; weighted vs unweighted head refit (validation); threshold (val-A) | 3 epochs, lr 2e-5, `xlm-roberta-base` |
| `hybrid_knn` | threshold (val-A) | 50 neighbours per retriever, RRF k 60, e5 encoder |
| `kev`, `jev`, `laya` | threshold only (val-A of the 80-pair sample: 32 pairs, 64 rows) | everything else (zero-shot) |
| votes | k and `k_closed` (1 to M), soft thresholds (val-B); weights are the members' val-A micro-F1 | member set |

## How the `cordis_v0.5.0` neural members were built

`bilstm` and `transformer` were first trained end to end (`models/cordis_v0.4.0`, 256-token
inputs) and then refit with `scripts/train_all.py --refit-head-from` (command in the
[runbook](runbook.md#2-train)). Refitting keeps each saved encoder, vocabulary and label
order fixed and refits only the sigmoid classifier head on all audited training records,
here with 512-token inputs. Features are held in memory; no feature cache files are written.
It is classifier refitting, not encoder retraining.

`--loss-weighting sqrt_inverse_frequency` weights positives by
`sqrt((N - positives) / positives)`, clamped to `[1, --max-positive-weight]` (default 20) and
computed from training targets only; weighted validation loss uses the same weights for early
stopping, so losses from different weighting schemes are not directly comparable.
`--loss-weighting none` keeps the unweighted loss. In this mode `--batch-size` controls
classifier training and `--feature-batch-size` encoder inference.

The teammate who built the release refit a weighted and an unweighted candidate from the same
v0.4.0 encoders and selected on validation: the highest pooled validation micro-F1 with
macro-F1 at least the v0.4.0 reference. Both families kept the unweighted refit. Each
artifact's `naltra.json` records the criterion, the candidates' scores and report hashes
(`metadata.validation_selection`), the hashes of the v0.4.0 artifact it started from
(`metadata.initial_artifact`), and both training histories (`metadata.initial_training_history`
and `history`). The candidates' own artifacts are not available. Validation was used for
checkpoint, candidate and threshold selection; the benchmark retunes member thresholds on
val-A and reports test scores only from `evaluate_ensemble.py`.

## Code-switching evaluation

The sentence/chunk tracks are synthetic evaluation-only variants built from aligned
EN/TR versions of the same project. Nothing is tuned on mixed text: member and
ensemble settings come from clean EN/TR validation.

Scores come from `cs_sentence_test` and `cs_chunk_test` dumps. Cross-condition
bootstrap comparisons use their shared `pair_id`. Kev, Jev, and Laya are not
evaluated on these tracks in the reported experiment.

## Sample protocol (Kev, Jev, Laya)

Kev, Jev and Laya are scored only on fixed pair-hash samples of 80 validation and 120 test
pairs (`scripts/sample_sets.py`, [runbook](runbook.md#kev-jev-and-laya-on-fixed-samples)).
Every trained member is rescored on the same files, and the run
(`results/metrics/cordis_v1.1.0_sample/`) is a benchmark of its own:

- Six members vote (`svm`, `hybrid_knn`, `bilstm`, `transformer`, `kev`, `jev`);
  `naive_bayes` and `laya` are scored but never vote (the summary's `baselines` key; Laya's
  role stays `new`). The sample's votes are therefore different systems from the
  four-member full-set votes.
- Member thresholds and weights are re-tuned on 32 val-A pairs (64 rows) and the vote
  settings on 48 val-B pairs (96 rows), so they differ from the full-set ones (SVM 0.12
  vs 0.16).
- Kev, Jev and Laya ask English questions about English label names for Turkish input too,
  so their EN -> TR drop mixes the input shift with cross-lingual prompting.
- Observed pairwise micro-F1 intervals have average half-widths of about 0.043 EN and 0.041
  TR (`pairwise.json`, `micro_f1_ci_half_width`); small differences remain uncertain, and an
  interval that includes 0 does not establish equivalence. The pairwise results are under
  [findings](results.md#findings).
- Jev answers in steps of 0.01, stored as float32. Thresholds are applied to float64-cast
  scores (`scripts/evaluate_ensemble.py`), and float32(0.88) is just below 0.88, so Jev's
  tuned 0.88 selects scores of 0.89 and above.
- Laya's tuned threshold, 0.98, is the top of the grid, where its val-A micro-F1 is 0.064, so
  a better threshold may lie above the grid. Its test scores, 0.068 EN / 0.046 TR micro-F1
  (EN -> TR drop 0.021 [-0.005, 0.048], Holm p 0.11, not significant), are a negative result
  for this zero-shot setup. No full-validation Laya number is reported.

## Recorded outputs

`evaluate_ensemble.py` records thresholds, vote settings, label/taxonomy identity,
input and artifact fingerprints, environment, code commit (`git describe --always
--dirty`), and bootstrap seed. `run_ood.py` records its filtered counts, model
identities, environment, commit, and bootstrap seeds. `scripts/extra_analyses.py`
adds label coverage, the sample's zero-shot pairwise tests, an error analysis, the
SVM drop tests and the Turkish translation screens, and, when named, language ID and
round-trip chrF (`mtquality`), each with its code commit and input hashes
([command](runbook.md#extra-analyses)). Result JSONs store repository-relative paths.

The full-set, sample and OOD summaries were written at commits `a9b3c1e`, `97f7e08`
and `cdf5f74`. Provenance gaps:

- `results/benchmarks/cordis_v1.1.0/benchmark_summary.json` predates the top-level
  `code_commit` and `environment` fields that `run_benchmark.py` now writes; it records
  the runtime and hardware, with the CPU as `Intel64 Family 6 Model 158 Stepping 10`
  (the Intel Core i5-9300H).
- The neural training histories (BiLSTM stopped after epoch 9 and kept epoch 7; XLM-R
  validation loss 0.0382 to 0.0345 over its 3 epochs) are in
  `metadata.initial_training_history` of the `cordis_v0.5.0` artifacts, which are not in Git.
- The Naive Bayes alpha sweep exists only as the comment in `configs/models/naive_bayes.yaml`.
  The near-duplicate counts and the cleaning-loss effect in the
  [results limitations](results.md#limitations) were computed from local data and
  prediction dumps; no committed script reproduces them.
- The label-universe check (on train and validation, 5 of the 473 labels have fewer than
  43 projects, the 50-project threshold scaled to those splits, and 54 fewer than 50), the
  near-OOD counts above, the 11 single-sentence documents per code-switch track and the
  realised noise corruption rates were computed from the committed data (the noise rates
  from regenerated noisy copies); no committed script reproduces them.
- The Kev server commit was not recorded, `jev-latest` is a moving hosted alias, and the
  `xlm-roberta-base` base revision is null
  ([training provenance](reproducibility.md#training-and-data-provenance)).
- Kev and Laya per-document times ([members](runbook.md#1-members),
  [rerun inference](reproducibility.md#rerun-inference)) come from local dump sidecars and a
  teammate's run; no committed file records them.

No values should be inserted manually to stand in for a missing experiment.
See [result files](../results/README.md), [runbook commands](runbook.md#4-benchmark),
and [findings](results.md).
