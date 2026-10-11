# Evaluation plan

All model families are evaluated through the same `PredictionResult` schema and fixed dataset splits.

## Areas

- Classification: precision, recall, micro-F1, macro-F1, and confusion matrices.
- Hierarchy: ancestor consistency and hierarchy-aware precision/recall/F1 using the 586-node EuroSciVoc taxonomy.
- Language: English, machine-translated Turkish and code-switched test sets of the same projects; every trained member sees EN+TR in training, so none is a cross-lingual transfer slice.
- Calibration: expected calibration error (ECE), Brier score, and reliability plots.
- OOD: detection AUROC and false-positive rate at 95% true-positive rate, with the humanities root held out of training (`scripts/run_ood.py`).
- Robustness: performance deltas on the noisy and code-switched test sets, with paired bootstrap tests across conditions and of SVM's drop against each other member's.
- Efficiency: warmed-up latency distributions under documented hardware and batch sizes.
- Ensemble comparison: hard, soft, and weighted-soft voting against each voting member (`svm`, `hybrid_knn`, `bilstm`, `transformer`); `naive_bayes` is reported but does not vote, because it uses the same TF-IDF features as `svm` and would give that representation a second vote; Kev and Jev are compared with every other member, and vote with them, on the fixed pair-hash sample (80 validation, 120 test pairs); Laya is scored on the same sample as a non-voting member (0.068 EN / 0.046 TR micro-F1).

Benchmark scripts must save their configuration, taxonomy version, dataset checksums, random seeds, and environment details. `evaluate_ensemble.py` writes them into `summary.json`: `code_commit` (`git describe --always --dirty`), `environment`, `bootstrap_seed`, the SHA-256 of every input set, dump and artifact, and the voting settings (member thresholds, the val-A micro-F1 used as weights, vote counts `k` and `k_closed`, soft thresholds). `run_ood.py` and `run_benchmark.py` record `code_commit` and `environment` as well (the committed `benchmark_summary.json` predates that field, see [Provenance](#provenance)), and the OOD summary its bootstrap seeds. Result JSONs store repository-relative paths. Never add invented or manually filled experimental values; generated tables belong under `results/`.

CORDIS ordinary classification compares `labels_direct` against direct predictions over
the fixed supported label universe. Hierarchical metrics expand ancestors for both sides.
Gold closures start from the 473 retained labels only: CORDIS codes with fewer than 50
supporting projects are dropped, not rolled up to a supported ancestor. Within the active
586-node taxonomy, 1,039 of the 4,711 test projects lack 2,375 CORDIS-implied nodes.
Restoring those active nodes changes hierarchical F1 by at most 0.007 and no ranking; full
unpruned-tree performance was not evaluated (`results/metrics/label_coverage.json`).
English, translated Turkish, and synthetic code-switch records inherit their source project split.

## Benchmark and metric choice

`configs/benchmark.yaml` fixes the primary metric (micro-F1), the secondary table columns and,
for every system, its role (course baseline or new method), architecture and model.
`scripts/evaluate_ensemble.py` copies these into `summary.json["benchmark"]`, refuses any scored
system without an entry, and opens `summary.md` with a `## Benchmark` section: per test set,
every system ranked by micro-F1 (leave-one-out rows carry micro-F1 only; `hard_closed` has
only hierarchical scores and is listed unranked), then the label distribution of the tuning
set and the first test set.

### The label distribution

`naltra.evaluation.matrix.label_distribution` over `labels_direct` and the fixed 473-label
universe (the dumps' label columns), computed from the repository root with:

```bash
python - <<'EOF'
import json
from naltra.data.loader import load_jsonl
from naltra.evaluation.matrix import binarize, label_distribution
taxonomy = json.load(open("taxonomy/taxonomy.json", encoding="utf-8"))["labels"]
labels = sorted(item["id"] for item in taxonomy if item.get("is_direct_supported"))
for split in ("test", "validation"):
    rows = load_jsonl(f"data/processed/cordis_h2020/en/{split}.jsonl")
    print(split, label_distribution(binarize([r["labels_direct"] for r in rows], labels)))
EOF
```

| statistic | EN test | EN validation |
|---|---|---|
| records / labels | 4,711 / 473 | 4,711 / 473 |
| labels per record, mean / median | 3.216 / 3 | 3.231 / 3 |
| label prevalence, min / median / max | 0.00127 / 0.00403 / 0.0565 | 0.00127 / 0.00403 / 0.0563 |
| labels under 1% prevalence | 386 (81.6%) | 386 (81.6%) |
| positives carried by the most frequent 10% of labels | 36.4% | 36.3% |
| max / min prevalence (labels with a positive) | 44.3 | 44.2 |
| labels with no positive | 0 | 0 |
| per-label accuracy of predicting no label | 0.9932 | 0.9932 |

The median label occurs in 19 of 4,711 test projects. Turkish records are translations of
the same projects with the same labels, so the TR sets have the same distribution. The
fixed Kev/Jev samples cover far fewer labels: 249 of the 473 have no positive project in
the 120-pair test sample and 304 none in the 80-pair validation sample
(`zero_positive_labels` in `summary.json["benchmark"]["label_distribution"]` of the sample
run, `results/metrics/cordis_v1.1.0_sample/`), so sample
macro-F1 cannot exceed 224/473 = 0.474 and is not comparable to these full sets.

### Why not accuracy

- Per-label (Hamming) accuracy is dominated by true negatives: predicting no label at all
  scores 0.9932, so real systems differ only in the third decimal, and a system that
  misses every rare label loses almost nothing.
- Subset (exact-set) accuracy needs all of about 3.2 labels out of 473 to be exactly right.
  It gives no credit for two correct labels out of three or for a sibling of the right
  field, so it cannot separate systems that differ in how much they get right.

### What each metric adds

- **Micro-F1 (primary).** Pools true positives, false positives and false negatives over
  every (project, label) cell and ignores the true negatives that make accuracy
  uninformative. It weights labels by how often they occur, which matches the task of
  tagging each project with its fields, and it is the objective every threshold, the
  vote count k and the soft thresholds are tuned for on validation, so the ranking metric
  and the tuning objective agree.
- **Macro-F1.** Each of the 473 labels counts equally, absent labels as zero, so it
  exposes the 386-label rare tail that micro-F1 underweights. It is noisier (19 test
  positives for the median label), hence secondary.
- **Hierarchical micro-F1 (`hmicro_f1`).** Gold and predicted sets are closed upward over
  the 586-node EuroSciVoc tree before scoring, so a wrong leaf under the right field earns
  partial credit; it is the only comparable score for the hierarchy-closed vote.
- **Ranking metrics (P@1/3/5, R-precision, nDCG@5).** Threshold-free: they judge the score
  order regardless of the tuned threshold. P@1 asks whether the top label is correct.
- **Calibration (ECE, top-1 ECE, Brier).** Whether probabilities mean what they say,
  which soft voting and OOD thresholds rely on. Pooled ECE over all cells is dominated by
  near-zero negatives, so the benchmark table shows top-1 ECE.
- **OOD AUROC and FPR@95%TPR** (`scripts/run_ood.py`, held-out humanities). AUROC ranks
  in- versus out-of-distribution inputs without a threshold; FPR@95%TPR is the share of
  in-distribution projects wrongly flagged when 95% of OOD inputs are caught.
- **Paired project-grouped bootstrap with Holm.** 10,000 resamples of `pair_id`, so the
  EN, TR and code-switch variants of one project count once; 95% percentile intervals of
  the micro- and macro-F1 differences. The p-value is the two-sided bootstrap sign
  proportion with an add-one correction, `min(1, 2(1 + min(#d <= 0, #d >= 0)) / (B + 1))`
  over the B resampled differences d, so it is never below 2/10,001
  (`src/naltra/evaluation/bootstrap.py`). The benchmark table reports point estimates; the
  intervals are in `## Significance`. On sparse-label samples a percentile interval for a
  macro-F1 difference can sit outside its point estimate, because resamples omit rare
  labels (the 473-label denominator and the zero-division rule are kept), so read those
  intervals cautiously.

Per-label ROC-AUC is not reported: with a median prevalence of 0.4%, it is driven by the
many easy negatives, and the ranking metrics above already measure threshold-free quality.

### What is significance-tested

Holm adjusts each family separately, per metric:

| family | comparisons | where |
|---|---|---|
| votes vs best member, per test set | the 4 votes (`hard_majority`, `hard_k`, `soft`, `weighted_soft`) minus the member with the best val-A micro-F1 (`svm`) | `summary.json["significance"]["vs_best_member"]` |
| each system across conditions | first test set minus each other test set: 5 on the full sets (TR, both code-switch, both noisy), 1 on the sample (TR), where Holm therefore changes nothing | `summary.json["significance"]["cross_condition"]` |
| zero-shot vs trained members, per sample language | Kev, Jev and Laya each minus each of the 5 trained members (15 tests) | `results/metrics/cordis_v1.1.0_sample/pairwise.json` (`scripts/extra_analyses.py`) |
| SVM's drop vs each other member's, per condition (micro-F1 only) | SVM's EN-minus-condition drop minus that of `hybrid_knn`, `bilstm`, `transformer` and `naive_bayes` (4 tests) on `tr_test`, `cs_sentence_test`, `cs_chunk_test` and `en_test_noisy`; one shared resample of `pair_id` for both sets and all members; `tr_test_noisy` is not in this family | `results/metrics/cordis_v1.1.0_ensemble/robustness.json` (`scripts/extra_analyses.py drops`) |

All 16 SVM differences are above 0, each with a 95% interval excluding 0 and Holm p 0.0008,
the smallest that four tests over 10,000 resamples can give; they range from 0.030 [0.023,
0.038] (BiLSTM, noisy EN) to 0.067 [0.061, 0.074] (Naive Bayes, TR). The per-member drops in
the same file are the `cross_condition` ones (the script stops if a point estimate differs by
more than 1e-6).

The near-OOD AUROCs carry bootstrap intervals (`results/ood/cordis_v1.1.0/summary.json`)
but no test between models. Not tested at all, so only descriptive: differences between two
members on the full sets (for example XLM-R vs Naive Bayes), differences between two systems'
drops outside the SVM-minus-member family (for example `hybrid_knn`'s EN -> TR drop, 0.011, vs
Naive Bayes's, 0.002, any drop difference on noisy TR, or a vote's drop vs a member's),
relative drops, `hard_closed`, leave-one-out, calibration and latency.

### Baselines and new methods

Baselines are the course methods in their vanilla form; everything else is new. The
architecture is the method family, the model the concrete configured instance; for example,
the Transformer-encoder baseline is `xlm-roberta-base`, and a DeBERTa model would be a new
method.

| system | role | architecture | model |
|---|---|---|---|
| `naive_bayes` | baseline | multinomial Naive Bayes over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k, min_df 2, sublinear tf), one MultinomialNB per label, alpha 0.03 |
| `svm` | baseline | linear SVM over TF-IDF, one-vs-rest | same TF-IDF, one LinearSVC (C 1.0) per label, 3-fold project-grouped Platt calibration |
| `bilstm` | baseline | bidirectional LSTM | release `cordis_v0.5.0`: 1 layer, 300-d embeddings learned from scratch (50k vocabulary), 256 hidden per direction, dropout 0.2, trained with early stopping (patience 2, at most 10 epochs): it stopped after epoch 9 and kept the epoch-7 checkpoint, which had the lowest validation loss; then its sigmoid head refit on the frozen encoder (unweighted BCE, lr 1e-3); max_length 512 |
| `transformer` | baseline | Transformer encoder (RoBERTa family) | release `cordis_v0.5.0`: `xlm-roberta-base` fine-tuned 3 epochs (lr 2e-5), then its sigmoid head refit on the frozen encoder for up to 20 epochs (unweighted BCE, lr 1e-3); max_length 512 |
| `hybrid_knn` | new | sparse + dense retrieval kNN with reciprocal rank fusion (Cormack et al. 2009, k = 60) | TF-IDF cosine + `intfloat/multilingual-e5-base` embeddings, 50 neighbours per retriever, RRF k 60 |
| `kev` | new | zero-shot yes/no decision model (open-source Jev alternative) | Kev-0.8B (`jaredpalmer/kev-0.8b` at revision `bf75a6a8848ea6960ff2ed108d9ed44c2941174f`, served via the tag `v1.0`), p(yes) for "Is this project about {name}?" per label |
| `jev` | new | hosted System One decision model (Jev), zero-shot yes/no per label | TypeSafe `jev-latest`, asked the same "Is this project about {name}?" questions as Kev |
| `laya` | new | local pretrained multilingual choice-question model (Laya), zero-shot | `convaiinnovations/laya` multilingual checkpoint at revision `7b928d828b7b0e022f929d9bd2e44165aa270148` via laya SDK 0.4.1; one two-option question per direct label (A = the label's field name, B = a fixed "other" option), P(A) as the label score |
| `hard_majority`, `hard_k` | new | NALTRA hard vote | strict majority; k-of-M with k tuned on val-B |
| `hard_closed` | new | NALTRA hierarchy-closed hard vote | k-of-M over ancestor-closed 586-node label sets |
| `soft`, `weighted_soft` | new | NALTRA soft vote | mean, or val-A micro-F1-weighted mean, of raw member scores (each on its own scale, not recalibrated); threshold tuned on val-B |
| `leave_one_out` | new | NALTRA hard vote (ablation) | strict majority of the members left after dropping each one |

`hybrid_knn`, `kev`, `jev` and `laya` are the new single-model methods: `hybrid_knn` is a
sparse+dense retrieval kNN that fuses the TF-IDF and multilingual-e5 neighbour rankings by
reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009; the standard k = 60),
`kev` is an open-weights Jev-style decision model queried with one yes/no question per
label, the `jev` adapter scores the hosted service with the same questions, and `laya` asks
a pinned local multilingual Laya checkpoint one two-option question per label
([core ML](core_ml.md#local-multilingual-laya)). NALTRA's votes over all members are the
project's own new method. `hybrid_knn` is unrelated to the `laya` member: the comment
"Laya uses k=60" in `configs/models/hybrid_knn.yaml` refers only to the RRF constant.

### Tuning budget

What each system searched, and on which data. "Validation" is the full EN+TR validation set
(both halves); thresholds come from the 49-value grid 0.02-0.98 (`THRESHOLD_GRID` in
`src/naltra/evaluation/offline.py`). Every trained member is one run with seed 42.

| system | searched (selection data) | fixed, not searched |
|---|---|---|
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
and `history`). The candidates' own artifacts are not available. Validation was used for checkpoint,
candidate and threshold selection; the benchmark retunes member thresholds on val-A and
reports test scores only from `evaluate_ensemble.py`.

## Code-switching evaluation

Code-switching is an evaluation-only track built from the audited EN/TR versions of the
same projects: `sentence_mix` and `chunk_mix`, balanced, 4,711 validation and 4,711 test
records each ([dataset specification](dataset.md#9-code-switch-benchmark-track)). They are synthetic
mixed-language variants, not collected bilingual documents; no model is trained on them and
nothing is translated again. The shipped tracks are restored with the rest of the release by
`scripts/unpack_data.py`.

The reported code-switch scores are produced like every other test set. `predict_all.py`
dumps each locally trained member's scores on `cs_chunk_test` and `cs_sentence_test`
(`data/processed/cordis_h2020/code_switch/{chunk,sentence}_mix/balanced/test.jsonl`), and
`evaluate_ensemble.py` scores the members and the votes on them with the member thresholds,
vote counts and soft thresholds tuned on clean EN+TR validation (val-A and val-B); nothing is
tuned on code-switch data. The code-switch rows carry the `pair_id` of their source project,
so the cross-condition bootstrap in `## Significance` compares each system on the first test
set with each code-switch set, counting every project once. Kev, Jev and Laya are not scored
on the code-switch sets; they run only on the fixed 80/120-pair samples. The commands are in the
[runbook](runbook.md#3-dump-predictions).

## Sample protocol (Kev, Jev, Laya)

Kev, Jev and Laya are scored only on fixed pair-hash samples of 80 validation and 120 test
pairs (`scripts/sample_sets.py`, [runbook](runbook.md#kev-jev-and-laya-on-fixed-samples)).
Every other member is rescored on the same files, and the run
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
  interval that includes 0 does not establish equivalence. On English, Jev and Kev cannot be
  separated from XLM-R, BiLSTM or Naive Bayes (Jev - XLM-R +0.012 [-0.033, +0.058], Jev -
  Naive Bayes -0.020 [-0.062, +0.022], Kev - XLM-R +0.000 [-0.046, +0.048]). On Turkish,
  Jev - XLM-R is +0.037 [-0.006, +0.080], but Kev is below Naive Bayes (-0.078 [-0.115,
  -0.041], Holm p 0.003) and BiLSTM (-0.058 [-0.099, -0.015], Holm p 0.045). Laya is below
  every trained member in both languages (Holm p 0.003) (`pairwise.json`).
- Jev answers in steps of 0.01, stored as float32. Thresholds are applied to float64-cast
  scores (`scripts/evaluate_ensemble.py`), and float32(0.88) is just below 0.88, so Jev's
  tuned 0.88 selects scores of 0.89 and above.
- Laya's tuned threshold, 0.98, is the top of the grid, where its val-A micro-F1 is 0.064, so
  a better threshold may lie above the grid. Its test scores, 0.068 EN / 0.046 TR micro-F1
  (EN -> TR drop 0.021 [-0.005, 0.048], Holm p 0.11, not significant), are a negative result
  for this zero-shot setup. No full-validation Laya number is reported.

## Provenance

The full-set, sample and OOD summaries record the commits they were written at (`a9b3c1e`,
`97f7e08` and `cdf5f74`). Gaps:

- `results/benchmarks/cordis_v1.1.0/benchmark_summary.json` predates the `code_commit`
  field; it records the runtime and hardware, with the CPU as `Intel64 Family 6 Model 158
  Stepping 10` (the Intel Core i5-9300H named in the docs).
- The neural training histories (BiLSTM stopped after epoch 9 and kept epoch 7; XLM-R
  validation loss 0.0382 to 0.0345 over its 3 epochs) are in
  `metadata.initial_training_history` of the `cordis_v0.5.0` artifacts, which are not in Git.
- The Naive Bayes alpha sweep exists only as the comment in `configs/models/naive_bayes.yaml`;
  the near-duplicate counts and the cleaning-loss effect in the README's Limitations were
  computed from local data and prediction dumps, and no committed script reproduces them.
- The label-universe counts in the README's Limitations (on train and validation, 5 of 473
  labels below a scaled threshold of 43 projects, 54 below 50), the near-OOD counts of projects
  with only a sub-threshold humanities code (165 train, 19 validation, 34 in-domain test;
  [dataset card](../data/README.md#known-issues), item 4), the 11 single-sentence documents
  per code-switch track ([dataset specification](dataset.md#9-code-switch-benchmark-track))
  and the realised noise corruption rates
  ([section 10](dataset.md#10-noisy-test-track-generated-not-shipped)) were computed from
  the committed data (the noise rates from regenerated noisy copies), and no committed
  script reproduces them.
- The Kev server commit was not recorded, `jev-latest` is a moving hosted alias, and the
  `xlm-roberta-base` base revision is null ([runbook](runbook.md#known-limitations)).
