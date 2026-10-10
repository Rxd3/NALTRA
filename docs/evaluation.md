# Evaluation plan

All model families are evaluated through the same `PredictionResult` schema and fixed dataset splits.

## Areas

- Classification: precision, recall, micro-F1, macro-F1, and confusion matrices.
- Hierarchy: ancestor consistency and hierarchy-aware precision/recall/F1 using the 586-node EuroSciVoc taxonomy.
- Language: English, Turkish, code-switched, and cross-language transfer slices.
- Calibration: expected calibration error (ECE), Brier score, and reliability plots.
- OOD: detection AUROC and false-positive rate at 95% true-positive rate, with the humanities root held out of training (`scripts/run_ood.py`).
- Robustness: performance deltas on the noisy and code-switched test sets, with paired bootstrap tests across conditions.
- Efficiency: warmed-up latency distributions under documented hardware and batch sizes.
- Ensemble comparison: hard, soft, and weighted-soft voting against each voting member (`svm`, `hybrid_knn`, `bilstm`, `transformer`); `naive_bayes` is reported but does not vote, because it uses the same TF-IDF features as `svm` and would give that representation a second vote; Kev and Jev are compared with every other member, and vote with them, on the fixed pair-hash sample (80 validation, 120 test pairs); Laya is scored on the same sample as a non-voting member (results pending).

Benchmark scripts must save their configuration, taxonomy version, dataset checksums, random seeds, and environment details. `evaluate_ensemble.py` writes them into `summary.json`: `code_commit` (`git describe --always --dirty`), `environment`, `bootstrap_seed`, the SHA-256 of every input set, dump and artifact, and the voting settings (member thresholds, the val-A micro-F1 used as weights, vote counts `k` and `k_closed`, soft thresholds). `run_ood.py` and `run_benchmark.py` record `code_commit` and `environment` as well, and the OOD summary its bootstrap seeds. Result JSONs store repository-relative paths. Never add invented or manually filled experimental values; generated tables belong under `results/`.

CORDIS ordinary classification compares `labels_direct` against direct predictions over
the fixed supported label universe. Hierarchical metrics expand ancestors for both sides.
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
  the micro- and macro-F1 differences; Holm controls the family-wise error across each
  family of comparisons (ensembles vs the best member, each system across conditions).
  The benchmark table reports point estimates; the intervals are in `## Significance`.
  On sparse-label samples a percentile interval for a macro-F1 difference can sit outside
  its point estimate, because resamples omit rare labels (the 473-label denominator and the
  zero-division rule are kept), so read those intervals cautiously.

Per-label ROC-AUC is not reported: with a median prevalence of 0.4%, it is driven by the
many easy negatives, and the ranking metrics above already measure threshold-free quality.

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
| `laya` | new | local pretrained multilingual choice-question model (Laya), zero-shot | `convaiinnovations/laya` multilingual checkpoint at revision `7b928d828b7b0e022f929d9bd2e44165aa270148` via laya SDK 0.4.1; one two-option question per direct label (A = the topic, B = other research topics), P(A) as the label score |
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
project's own new method.

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

Code-switching is an evaluation-only track built from the audited EN/TR versions of the
same projects: `sentence_mix` and `chunk_mix`, balanced, 4,711 validation and 4,711 test
records each ([dataset card](dataset.md#9-code-switch-benchmark-track)). They are synthetic
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
