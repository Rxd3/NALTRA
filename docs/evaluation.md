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
representation as SVM. All five trained models see EN+TR training text.

Kev and Jev are evaluated with the same five models on fixed 80-validation/120-test
project-pair samples. Their validation sample splits into 32 val-A pairs and 48
val-B pairs. Kev/Jev join the sample votes. Laya is a non-voting sample baseline with recorded micro-F1 of
0.068 English / 0.046 Turkish (threshold 0.98, tuned on val-A).

Member hyperparameters, neural checkpoints, and head-refit candidates were selected
using the full validation set before this val-A/val-B protocol. This validation
reuse is a limitation; test sets were only scored.

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

## Uncertainty and significance

The benchmark uses 10,000 paired bootstrap resamples grouped by `pair_id`,
95% percentile intervals, and Holm correction within each comparison family.
It compares ensembles with the best member selected on val-A and each system across
input conditions. Results are in each summary's `significance` block.

Project pairing keeps language variants together in resampling. Intervals describe
test-set sampling uncertainty for the recorded training run; they do not measure
training variance. Sparse sample macro-F1 intervals need care because bootstrap
samples can omit rare labels while retaining the fixed denominator.

## OOD and efficiency

The near-OOD experiment removes projects touching the humanities root from both
training and validation, then retrains NB and SVM. Test projects with only
humanities labels are OOD; projects with no humanities label are in-distribution.
Mixed projects are excluded.

Per language, this gives 4,351 in-distribution, 94 OOD, and 266 excluded projects.
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
and `history`). The candidate artifacts are not shipped. Validation was used for checkpoint,
candidate and threshold selection; the benchmark retunes member thresholds on val-A and
reports test scores only from `evaluate_ensemble.py`.



## Code-switching evaluation

The sentence/chunk tracks are synthetic evaluation-only variants built from aligned
EN/TR versions of the same project. Nothing is tuned on mixed text: member and
ensemble settings come from clean EN/TR validation.

Scores come from `cs_sentence_test` and `cs_chunk_test` dumps. Cross-condition
bootstrap comparisons use their shared `pair_id`. Kev, Jev, and Laya are not
evaluated on these tracks in the reported experiment.

## Recorded outputs

`evaluate_ensemble.py` records thresholds, vote settings, label/taxonomy identity,
input and artifact fingerprints, environment, code commit, and bootstrap seed.
`run_ood.py` records its filtered counts, model identities, environment, commit,
and bootstrap seeds. The latency producer records runtime metadata; the archived
latency summary lacks current top-level code/environment fields.

No values should be inserted manually to stand in for a missing experiment.
See [result files](../results/README.md), [runbook commands](runbook.md#4-benchmark),
and [findings](results.md).
