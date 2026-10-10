## Benchmark

Ranked by micro_f1 (ties share a rank); ablations and systems without it are unranked. Paired-difference intervals between systems and conditions: see Significance.

### en_test_kev

| rank | system | role | architecture | model | micro_f1 | macro_f1 | hmicro_f1 | p_at_1 | top1_ece |
|---|---|---|---|---|---|---|---|---|---|
| 1 | svm | baseline | linear SVM over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k features, min_df 2, sublinear tf) + one LinearSVC (C 1.0) per label, 3-fold project-grouped Platt calibration | 0.5557 | 0.2541 | 0.6659 | 0.8667 | 0.1221 |
| 2 | weighted_soft | new | NALTRA weighted soft vote | member probabilities weighted by val-A micro-F1, global threshold tuned on val-B | 0.5096 | 0.2066 | 0.6219 | 0.8083 | 0.2246 |
| 3 | soft | new | NALTRA soft vote | mean member probability, global threshold tuned on val-B | 0.4638 | 0.1703 | 0.5768 | 0.7417 | 0.1792 |
| 4 | hard_k | new | NALTRA hard vote | k-of-M vote of member label sets, k tuned on val-B for micro-F1 | 0.4496 | 0.1770 | 0.5790 | - | - |
| 5 | hard_majority | new | NALTRA hard vote | strict majority (more than M/2) of member label sets, member thresholds tuned on val-A | 0.4105 | 0.1466 | 0.5228 | - | - |
| 6 | hybrid_knn | new | sparse + dense retrieval kNN with reciprocal rank fusion that reimplements Laya's retrieval design (Laya itself is not run) | TF-IDF 1-2-grams (50k) cosine + intfloat/multilingual-e5-base mean-pooled embeddings (max_length 256), 50 neighbours per retriever, RRF k 60, fused neighbour label vote | 0.3770 | 0.1554 | 0.5406 | 0.6417 | 0.2915 |
| 7 | bilstm | baseline | bidirectional LSTM | release cordis_v0.5.0: 1-layer BiLSTM trained from scratch (300-d embeddings, 50k vocabulary, 256 hidden units per direction, dropout 0.2) with early stopping, which ended training after epoch 9 and kept the lowest-validation-loss epoch-7 checkpoint; then its sigmoid head refit on the frozen encoder (unweighted BCE, lr 1e-3); max_length 512 | 0.2983 | 0.1081 | 0.4507 | 0.5083 | 0.1362 |
| 8 | naive_bayes | baseline | multinomial Naive Bayes over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k features, min_df 2, sublinear tf) + one binary MultinomialNB per label, alpha 0.03 | 0.2967 | 0.1342 | 0.4596 | 0.5500 | 0.2509 |
| 9 | jev | new | hosted System One decision model (Jev), zero-shot yes/no per label | TypeSafe jev-latest via POST /v1/systemone, one noul question per direct label ("Is this project about {name}?"), no CORDIS fine-tuning by the team | 0.2771 | 0.1532 | 0.4925 | 0.3833 | 0.5878 |
| 10 | kev | new | zero-shot yes/no decision model (open-source Jev alternative) | Kev-0.8B: jaredpalmer/kev-0.8b at bf75a6a8848ea6960ff2ed108d9ed44c2941174f (tag v1.0) on Qwen/Qwen3.5-0.8B-Base, System One server; one yes/no ("noul") question per direct label, p(yes) for "Is this project about {name}?"; not fine-tuned on CORDIS by the team | 0.2653 | 0.1586 | 0.4718 | 0.3333 | 0.6107 |
| 11 | transformer | baseline | Transformer encoder (RoBERTa family) | release cordis_v0.5.0: xlm-roberta-base fine-tuned for 3 epochs (lr 2e-5), then its sigmoid head refit on the frozen encoder for up to 20 epochs (unweighted BCE, lr 1e-3); max_length 512 | 0.2649 | 0.0911 | 0.4594 | 0.3833 | 0.0725 |
| - | hard_closed | new | NALTRA hierarchy-closed hard vote | k-of-M vote over ancestor-closed 586-node label sets, k tuned on val-B | - | - | 0.6126 | - | - |
| - | leave_one_out(-svm) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.3883 | - | - | - | - |
| - | leave_one_out(-hybrid_knn) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.4404 | - | - | - | - |
| - | leave_one_out(-bilstm) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.4435 | - | - | - | - |
| - | leave_one_out(-transformer) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.4464 | - | - | - | - |
| - | leave_one_out(-kev) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.4397 | - | - | - | - |
| - | leave_one_out(-jev) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.4356 | - | - | - | - |

### tr_test_kev

| rank | system | role | architecture | model | micro_f1 | macro_f1 | hmicro_f1 | p_at_1 | top1_ece |
|---|---|---|---|---|---|---|---|---|---|
| 1 | svm | baseline | linear SVM over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k features, min_df 2, sublinear tf) + one LinearSVC (C 1.0) per label, 3-fold project-grouped Platt calibration | 0.4893 | 0.2157 | 0.6189 | 0.7917 | 0.1391 |
| 2 | weighted_soft | new | NALTRA weighted soft vote | member probabilities weighted by val-A micro-F1, global threshold tuned on val-B | 0.4678 | 0.1869 | 0.5838 | 0.7750 | 0.2121 |
| 3 | soft | new | NALTRA soft vote | mean member probability, global threshold tuned on val-B | 0.4281 | 0.1617 | 0.5379 | 0.7250 | 0.1735 |
| 4 | hard_k | new | NALTRA hard vote | k-of-M vote of member label sets, k tuned on val-B for micro-F1 | 0.4239 | 0.1685 | 0.5580 | - | - |
| 5 | hard_majority | new | NALTRA hard vote | strict majority (more than M/2) of member label sets, member thresholds tuned on val-A | 0.3973 | 0.1376 | 0.5147 | - | - |
| 6 | hybrid_knn | new | sparse + dense retrieval kNN with reciprocal rank fusion that reimplements Laya's retrieval design (Laya itself is not run) | TF-IDF 1-2-grams (50k) cosine + intfloat/multilingual-e5-base mean-pooled embeddings (max_length 256), 50 neighbours per retriever, RRF k 60, fused neighbour label vote | 0.3821 | 0.1563 | 0.5429 | 0.6167 | 0.2782 |
| 7 | naive_bayes | baseline | multinomial Naive Bayes over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k features, min_df 2, sublinear tf) + one binary MultinomialNB per label, alpha 0.03 | 0.2980 | 0.1224 | 0.4634 | 0.5167 | 0.2094 |
| 8 | jev | new | hosted System One decision model (Jev), zero-shot yes/no per label | TypeSafe jev-latest via POST /v1/systemone, one noul question per direct label ("Is this project about {name}?"), no CORDIS fine-tuning by the team | 0.2786 | 0.1474 | 0.4937 | 0.3500 | 0.6178 |
| 9 | bilstm | baseline | bidirectional LSTM | release cordis_v0.5.0: 1-layer BiLSTM trained from scratch (300-d embeddings, 50k vocabulary, 256 hidden units per direction, dropout 0.2) with early stopping, which ended training after epoch 9 and kept the lowest-validation-loss epoch-7 checkpoint; then its sigmoid head refit on the frozen encoder (unweighted BCE, lr 1e-3); max_length 512 | 0.2779 | 0.0933 | 0.4526 | 0.4750 | 0.1184 |
| 10 | transformer | baseline | Transformer encoder (RoBERTa family) | release cordis_v0.5.0: xlm-roberta-base fine-tuned for 3 epochs (lr 2e-5), then its sigmoid head refit on the frozen encoder for up to 20 epochs (unweighted BCE, lr 1e-3); max_length 512 | 0.2413 | 0.0776 | 0.4235 | 0.3417 | 0.0960 |
| 11 | kev | new | zero-shot yes/no decision model (open-source Jev alternative) | Kev-0.8B: jaredpalmer/kev-0.8b at bf75a6a8848ea6960ff2ed108d9ed44c2941174f (tag v1.0) on Qwen/Qwen3.5-0.8B-Base, System One server; one yes/no ("noul") question per direct label, p(yes) for "Is this project about {name}?"; not fine-tuned on CORDIS by the team | 0.2202 | 0.1473 | 0.4418 | 0.2500 | 0.7028 |
| - | hard_closed | new | NALTRA hierarchy-closed hard vote | k-of-M vote over ancestor-closed 586-node label sets, k tuned on val-B | - | - | 0.5899 | - | - |
| - | leave_one_out(-svm) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.3757 | - | - | - | - |
| - | leave_one_out(-hybrid_knn) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.3976 | - | - | - | - |
| - | leave_one_out(-bilstm) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.4300 | - | - | - | - |
| - | leave_one_out(-transformer) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.4292 | - | - | - | - |
| - | leave_one_out(-kev) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.4173 | - | - | - | - |
| - | leave_one_out(-jev) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.4225 | - | - | - | - |

### Label distribution

all_negative_accuracy is the per-label accuracy of predicting no label at all.

| statistic | tuning | en_test_kev |
|---|---|---|
| n_records | 160 | 120 |
| n_labels | 473 | 473 |
| mean_labels_per_record | 3.1625 | 3.1833 |
| median_labels_per_record | 3.5000 | 4.0000 |
| min_prevalence | 0.0000 | 0.0000 |
| median_prevalence | 0.0000 | 0.0000 |
| max_prevalence | 0.0750 | 0.0667 |
| rare_labels | 304 | 381 |
| rare_label_share | 0.6427 | 0.8055 |
| top_decile_positive_share | 0.5178 | 0.4241 |
| prevalence_ratio | 6.0000 | 8.0000 |
| zero_positive_labels | 304 | 249 |
| all_negative_accuracy | 0.9933 | 0.9933 |

## en_test_kev

| system | micro_f1 | macro_f1 | hmicro_f1 | p_at_1 |
|---|---|---|---|---|
| svm | 0.5557 | 0.2541 | 0.6659 | 0.8667 |
| hybrid_knn | 0.3770 | 0.1554 | 0.5406 | 0.6417 |
| bilstm | 0.2983 | 0.1081 | 0.4507 | 0.5083 |
| transformer | 0.2649 | 0.0911 | 0.4594 | 0.3833 |
| kev | 0.2653 | 0.1586 | 0.4718 | 0.3333 |
| jev | 0.2771 | 0.1532 | 0.4925 | 0.3833 |
| naive_bayes | 0.2967 | 0.1342 | 0.4596 | 0.5500 |
| hard_majority | 0.4105 | 0.1466 | 0.5228 | - |
| hard_k | 0.4496 | 0.1770 | 0.5790 | - |
| hard_closed | - | - | 0.6126 | - |
| soft | 0.4638 | 0.1703 | 0.5768 | 0.7417 |
| weighted_soft | 0.5096 | 0.2066 | 0.6219 | 0.8083 |

## tr_test_kev

| system | micro_f1 | macro_f1 | hmicro_f1 | p_at_1 |
|---|---|---|---|---|
| svm | 0.4893 | 0.2157 | 0.6189 | 0.7917 |
| hybrid_knn | 0.3821 | 0.1563 | 0.5429 | 0.6167 |
| bilstm | 0.2779 | 0.0933 | 0.4526 | 0.4750 |
| transformer | 0.2413 | 0.0776 | 0.4235 | 0.3417 |
| kev | 0.2202 | 0.1473 | 0.4418 | 0.2500 |
| jev | 0.2786 | 0.1474 | 0.4937 | 0.3500 |
| naive_bayes | 0.2980 | 0.1224 | 0.4634 | 0.5167 |
| hard_majority | 0.3973 | 0.1376 | 0.5147 | - |
| hard_k | 0.4239 | 0.1685 | 0.5580 | - |
| hard_closed | - | - | 0.5899 | - |
| soft | 0.4281 | 0.1617 | 0.5379 | 0.7250 |
| weighted_soft | 0.4678 | 0.1869 | 0.5838 | 0.7750 |

## Significance

Paired project bootstrap, 10000 resamples, 95% percentile intervals, Holm-adjusted p. Best member on val-A: svm.

| test set | ensemble vs best member | micro-F1 diff [CI] | p (Holm) |
|---|---|---|---|
| en_test_kev | hard_majority | -0.1452 [-0.1928, -0.0986] | 0.0007999 |
| en_test_kev | hard_k | -0.1061 [-0.1463, -0.0650] | 0.0007999 |
| en_test_kev | soft | -0.0919 [-0.1373, -0.0471] | 0.0007999 |
| en_test_kev | weighted_soft | -0.0461 [-0.0823, -0.0108] | 0.009399 |
| tr_test_kev | hard_majority | -0.0920 [-0.1358, -0.0490] | 0.0007999 |
| tr_test_kev | hard_k | -0.0653 [-0.1026, -0.0287] | 0.0018 |
| tr_test_kev | soft | -0.0612 [-0.1060, -0.0164] | 0.0144 |
| tr_test_kev | weighted_soft | -0.0215 [-0.0596, +0.0149] | 0.2638 |

| system | comparison | micro-F1 diff [CI] | p (Holm) |
|---|---|---|---|
| svm | en_test_kev->tr_test_kev | +0.0664 [+0.0326, +0.1004] | 0.0002 |
| hybrid_knn | en_test_kev->tr_test_kev | -0.0051 [-0.0273, +0.0176] | 0.6533 |
| bilstm | en_test_kev->tr_test_kev | +0.0204 [-0.0197, +0.0597] | 0.311 |
| transformer | en_test_kev->tr_test_kev | +0.0236 [-0.0097, +0.0572] | 0.1718 |
| kev | en_test_kev->tr_test_kev | +0.0451 [+0.0206, +0.0695] | 0.0007999 |
| jev | en_test_kev->tr_test_kev | -0.0015 [-0.0178, +0.0140] | 0.8493 |
| naive_bayes | en_test_kev->tr_test_kev | -0.0013 [-0.0290, +0.0264] | 0.9191 |
| hard_majority | en_test_kev->tr_test_kev | +0.0132 [-0.0196, +0.0443] | 0.42 |
| hard_k | en_test_kev->tr_test_kev | +0.0257 [-0.0066, +0.0584] | 0.1202 |
| soft | en_test_kev->tr_test_kev | +0.0357 [+0.0051, +0.0659] | 0.0238 |
| weighted_soft | en_test_kev->tr_test_kev | +0.0418 [+0.0160, +0.0685] | 0.0012 |
