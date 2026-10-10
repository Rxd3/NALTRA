## Benchmark

Ranked by micro_f1 (ties share a rank); ablations and systems without it are unranked. Paired-difference intervals between systems and conditions: see Significance.

### en_test

| rank | system | role | architecture | model | micro_f1 | macro_f1 | hmicro_f1 | p_at_1 | top1_ece |
|---|---|---|---|---|---|---|---|---|---|
| 1 | svm | baseline | linear SVM over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k features, min_df 2, sublinear tf) + one LinearSVC (C 1.0) per label, 3-fold project-grouped Platt calibration | 0.5584 | 0.4893 | 0.6683 | 0.7455 | 0.1061 |
| 2 | weighted_soft | new | NALTRA weighted soft vote | raw member scores weighted by val-A micro-F1, global threshold tuned on val-B | 0.5228 | 0.4426 | 0.6408 | 0.7077 | 0.2589 |
| 3 | soft | new | NALTRA soft vote | mean of raw member scores, global threshold tuned on val-B | 0.4975 | 0.4222 | 0.6279 | 0.6790 | 0.2756 |
| 4 | hard_k | new | NALTRA hard vote | k-of-M vote of member label sets, k tuned on val-B for micro-F1 | 0.4385 | 0.3451 | 0.5836 | - | - |
| 5 | hybrid_knn | new | sparse + dense retrieval kNN with reciprocal rank fusion (Cormack et al. 2009, k = 60) | TF-IDF 1-2-grams (50k) cosine + intfloat/multilingual-e5-base mean-pooled embeddings (max_length 256), 50 neighbours per retriever, RRF k 60, fused neighbour label vote | 0.3929 | 0.3121 | 0.5446 | 0.5672 | 0.2212 |
| 6 | hard_majority | new | NALTRA hard vote | strict majority (more than M/2) of member label sets, member thresholds tuned on val-A | 0.3699 | 0.2125 | 0.4977 | - | - |
| 7 | bilstm | baseline | bidirectional LSTM | release cordis_v0.5.0: 1-layer BiLSTM trained from scratch (300-d embeddings, 50k vocabulary, 256 hidden units per direction, dropout 0.2) with early stopping, which ended training after epoch 9 and kept the lowest-validation-loss epoch-7 checkpoint; then its sigmoid head refit on the frozen encoder (unweighted BCE, lr 1e-3); max_length 512 | 0.2891 | 0.1610 | 0.4606 | 0.4356 | 0.0317 |
| 8 | naive_bayes | baseline | multinomial Naive Bayes over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k features, min_df 2, sublinear tf) + one binary MultinomialNB per label, alpha 0.03 | 0.2777 | 0.1976 | 0.4397 | 0.4549 | 0.2728 |
| 9 | transformer | baseline | Transformer encoder (RoBERTa family) | release cordis_v0.5.0: xlm-roberta-base fine-tuned for 3 epochs (lr 2e-5), then its sigmoid head refit on the frozen encoder for up to 20 epochs (unweighted BCE, lr 1e-3); max_length 512 | 0.2476 | 0.1194 | 0.4382 | 0.3823 | 0.0209 |
| - | hard_closed | new | NALTRA hierarchy-closed hard vote | k-of-M vote over ancestor-closed 586-node label sets, k tuned on val-B | - | - | 0.5960 | - | - |
| - | leave_one_out(-svm) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.3458 | - | - | - | - |
| - | leave_one_out(-hybrid_knn) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.3899 | - | - | - | - |
| - | leave_one_out(-bilstm) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.4482 | - | - | - | - |
| - | leave_one_out(-transformer) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.4500 | - | - | - | - |

### tr_test

| rank | system | role | architecture | model | micro_f1 | macro_f1 | hmicro_f1 | p_at_1 | top1_ece |
|---|---|---|---|---|---|---|---|---|---|
| 1 | svm | baseline | linear SVM over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k features, min_df 2, sublinear tf) + one LinearSVC (C 1.0) per label, 3-fold project-grouped Platt calibration | 0.4886 | 0.4228 | 0.6152 | 0.6899 | 0.1141 |
| 2 | weighted_soft | new | NALTRA weighted soft vote | raw member scores weighted by val-A micro-F1, global threshold tuned on val-B | 0.4723 | 0.3890 | 0.6004 | 0.6678 | 0.2454 |
| 3 | soft | new | NALTRA soft vote | mean of raw member scores, global threshold tuned on val-B | 0.4553 | 0.3709 | 0.5940 | 0.6396 | 0.2591 |
| 4 | hard_k | new | NALTRA hard vote | k-of-M vote of member label sets, k tuned on val-B for micro-F1 | 0.4154 | 0.3149 | 0.5631 | - | - |
| 5 | hybrid_knn | new | sparse + dense retrieval kNN with reciprocal rank fusion (Cormack et al. 2009, k = 60) | TF-IDF 1-2-grams (50k) cosine + intfloat/multilingual-e5-base mean-pooled embeddings (max_length 256), 50 neighbours per retriever, RRF k 60, fused neighbour label vote | 0.3817 | 0.2946 | 0.5338 | 0.5576 | 0.2293 |
| 6 | hard_majority | new | NALTRA hard vote | strict majority (more than M/2) of member label sets, member thresholds tuned on val-A | 0.3399 | 0.1895 | 0.4688 | - | - |
| 7 | naive_bayes | baseline | multinomial Naive Bayes over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k features, min_df 2, sublinear tf) + one binary MultinomialNB per label, alpha 0.03 | 0.2753 | 0.1656 | 0.4327 | 0.4315 | 0.2073 |
| 8 | bilstm | baseline | bidirectional LSTM | release cordis_v0.5.0: 1-layer BiLSTM trained from scratch (300-d embeddings, 50k vocabulary, 256 hidden units per direction, dropout 0.2) with early stopping, which ended training after epoch 9 and kept the lowest-validation-loss epoch-7 checkpoint; then its sigmoid head refit on the frozen encoder (unweighted BCE, lr 1e-3); max_length 512 | 0.2708 | 0.1544 | 0.4428 | 0.3991 | 0.0196 |
| 9 | transformer | baseline | Transformer encoder (RoBERTa family) | release cordis_v0.5.0: xlm-roberta-base fine-tuned for 3 epochs (lr 2e-5), then its sigmoid head refit on the frozen encoder for up to 20 epochs (unweighted BCE, lr 1e-3); max_length 512 | 0.2398 | 0.1113 | 0.4278 | 0.3632 | 0.0212 |
| - | hard_closed | new | NALTRA hierarchy-closed hard vote | k-of-M vote over ancestor-closed 586-node label sets, k tuned on val-B | - | - | 0.5774 | - | - |
| - | leave_one_out(-svm) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.3325 | - | - | - | - |
| - | leave_one_out(-hybrid_knn) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.3596 | - | - | - | - |
| - | leave_one_out(-bilstm) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.4179 | - | - | - | - |
| - | leave_one_out(-transformer) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.4186 | - | - | - | - |

### cs_chunk_test

| rank | system | role | architecture | model | micro_f1 | macro_f1 | hmicro_f1 | p_at_1 | top1_ece |
|---|---|---|---|---|---|---|---|---|---|
| 1 | svm | baseline | linear SVM over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k features, min_df 2, sublinear tf) + one LinearSVC (C 1.0) per label, 3-fold project-grouped Platt calibration | 0.5124 | 0.4551 | 0.6384 | 0.7124 | 0.1170 |
| 2 | weighted_soft | new | NALTRA weighted soft vote | raw member scores weighted by val-A micro-F1, global threshold tuned on val-B | 0.5008 | 0.4329 | 0.6270 | 0.6863 | 0.2286 |
| 3 | soft | new | NALTRA soft vote | mean of raw member scores, global threshold tuned on val-B | 0.4807 | 0.4167 | 0.6155 | 0.6646 | 0.2559 |
| 4 | hard_k | new | NALTRA hard vote | k-of-M vote of member label sets, k tuned on val-B for micro-F1 | 0.4277 | 0.3367 | 0.5748 | - | - |
| 5 | hybrid_knn | new | sparse + dense retrieval kNN with reciprocal rank fusion (Cormack et al. 2009, k = 60) | TF-IDF 1-2-grams (50k) cosine + intfloat/multilingual-e5-base mean-pooled embeddings (max_length 256), 50 neighbours per retriever, RRF k 60, fused neighbour label vote | 0.3847 | 0.3023 | 0.5353 | 0.5680 | 0.2338 |
| 6 | hard_majority | new | NALTRA hard vote | strict majority (more than M/2) of member label sets, member thresholds tuned on val-A | 0.3577 | 0.2052 | 0.4901 | - | - |
| 7 | bilstm | baseline | bidirectional LSTM | release cordis_v0.5.0: 1-layer BiLSTM trained from scratch (300-d embeddings, 50k vocabulary, 256 hidden units per direction, dropout 0.2) with early stopping, which ended training after epoch 9 and kept the lowest-validation-loss epoch-7 checkpoint; then its sigmoid head refit on the frozen encoder (unweighted BCE, lr 1e-3); max_length 512 | 0.2818 | 0.1614 | 0.4545 | 0.4222 | 0.0192 |
| 8 | naive_bayes | baseline | multinomial Naive Bayes over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k features, min_df 2, sublinear tf) + one binary MultinomialNB per label, alpha 0.03 | 0.2788 | 0.1942 | 0.4414 | 0.4551 | 0.2647 |
| 9 | transformer | baseline | Transformer encoder (RoBERTa family) | release cordis_v0.5.0: xlm-roberta-base fine-tuned for 3 epochs (lr 2e-5), then its sigmoid head refit on the frozen encoder for up to 20 epochs (unweighted BCE, lr 1e-3); max_length 512 | 0.2467 | 0.1193 | 0.4379 | 0.3681 | 0.0287 |
| - | hard_closed | new | NALTRA hierarchy-closed hard vote | k-of-M vote over ancestor-closed 586-node label sets, k tuned on val-B | - | - | 0.5888 | - | - |
| - | leave_one_out(-svm) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.3415 | - | - | - | - |
| - | leave_one_out(-hybrid_knn) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.3807 | - | - | - | - |
| - | leave_one_out(-bilstm) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.4346 | - | - | - | - |
| - | leave_one_out(-transformer) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.4308 | - | - | - | - |

### cs_sentence_test

| rank | system | role | architecture | model | micro_f1 | macro_f1 | hmicro_f1 | p_at_1 | top1_ece |
|---|---|---|---|---|---|---|---|---|---|
| 1 | svm | baseline | linear SVM over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k features, min_df 2, sublinear tf) + one LinearSVC (C 1.0) per label, 3-fold project-grouped Platt calibration | 0.5080 | 0.4507 | 0.6338 | 0.7069 | 0.1150 |
| 2 | weighted_soft | new | NALTRA weighted soft vote | raw member scores weighted by val-A micro-F1, global threshold tuned on val-B | 0.5029 | 0.4305 | 0.6284 | 0.6846 | 0.2232 |
| 3 | soft | new | NALTRA soft vote | mean of raw member scores, global threshold tuned on val-B | 0.4816 | 0.4161 | 0.6159 | 0.6589 | 0.2473 |
| 4 | hard_k | new | NALTRA hard vote | k-of-M vote of member label sets, k tuned on val-B for micro-F1 | 0.4243 | 0.3312 | 0.5719 | - | - |
| 5 | hybrid_knn | new | sparse + dense retrieval kNN with reciprocal rank fusion (Cormack et al. 2009, k = 60) | TF-IDF 1-2-grams (50k) cosine + intfloat/multilingual-e5-base mean-pooled embeddings (max_length 256), 50 neighbours per retriever, RRF k 60, fused neighbour label vote | 0.3834 | 0.2999 | 0.5354 | 0.5634 | 0.2258 |
| 6 | hard_majority | new | NALTRA hard vote | strict majority (more than M/2) of member label sets, member thresholds tuned on val-A | 0.3585 | 0.2096 | 0.4932 | - | - |
| 7 | bilstm | baseline | bidirectional LSTM | release cordis_v0.5.0: 1-layer BiLSTM trained from scratch (300-d embeddings, 50k vocabulary, 256 hidden units per direction, dropout 0.2) with early stopping, which ended training after epoch 9 and kept the lowest-validation-loss epoch-7 checkpoint; then its sigmoid head refit on the frozen encoder (unweighted BCE, lr 1e-3); max_length 512 | 0.2836 | 0.1633 | 0.4563 | 0.4267 | 0.0238 |
| 8 | naive_bayes | baseline | multinomial Naive Bayes over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k features, min_df 2, sublinear tf) + one binary MultinomialNB per label, alpha 0.03 | 0.2800 | 0.1987 | 0.4414 | 0.4528 | 0.2606 |
| 9 | transformer | baseline | Transformer encoder (RoBERTa family) | release cordis_v0.5.0: xlm-roberta-base fine-tuned for 3 epochs (lr 2e-5), then its sigmoid head refit on the frozen encoder for up to 20 epochs (unweighted BCE, lr 1e-3); max_length 512 | 0.2445 | 0.1182 | 0.4356 | 0.3662 | 0.0313 |
| - | hard_closed | new | NALTRA hierarchy-closed hard vote | k-of-M vote over ancestor-closed 586-node label sets, k tuned on val-B | - | - | 0.5850 | - | - |
| - | leave_one_out(-svm) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.3408 | - | - | - | - |
| - | leave_one_out(-hybrid_knn) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.3779 | - | - | - | - |
| - | leave_one_out(-bilstm) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.4312 | - | - | - | - |
| - | leave_one_out(-transformer) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.4315 | - | - | - | - |

### en_test_noisy

| rank | system | role | architecture | model | micro_f1 | macro_f1 | hmicro_f1 | p_at_1 | top1_ece |
|---|---|---|---|---|---|---|---|---|---|
| 1 | svm | baseline | linear SVM over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k features, min_df 2, sublinear tf) + one LinearSVC (C 1.0) per label, 3-fold project-grouped Platt calibration | 0.4801 | 0.4074 | 0.5951 | 0.6837 | 0.1281 |
| 2 | weighted_soft | new | NALTRA weighted soft vote | raw member scores weighted by val-A micro-F1, global threshold tuned on val-B | 0.4586 | 0.3657 | 0.5758 | 0.6733 | 0.2939 |
| 3 | soft | new | NALTRA soft vote | mean of raw member scores, global threshold tuned on val-B | 0.4447 | 0.3470 | 0.5712 | 0.6500 | 0.3162 |
| 4 | hard_k | new | NALTRA hard vote | k-of-M vote of member label sets, k tuned on val-B for micro-F1 | 0.3993 | 0.2900 | 0.5383 | - | - |
| 5 | hybrid_knn | new | sparse + dense retrieval kNN with reciprocal rank fusion (Cormack et al. 2009, k = 60) | TF-IDF 1-2-grams (50k) cosine + intfloat/multilingual-e5-base mean-pooled embeddings (max_length 256), 50 neighbours per retriever, RRF k 60, fused neighbour label vote | 0.3740 | 0.2799 | 0.5268 | 0.5511 | 0.2353 |
| 6 | hard_majority | new | NALTRA hard vote | strict majority (more than M/2) of member label sets, member thresholds tuned on val-A | 0.2838 | 0.1447 | 0.3934 | - | - |
| 7 | naive_bayes | baseline | multinomial Naive Bayes over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k features, min_df 2, sublinear tf) + one binary MultinomialNB per label, alpha 0.03 | 0.2617 | 0.1568 | 0.4023 | 0.4341 | 0.1753 |
| 8 | bilstm | baseline | bidirectional LSTM | release cordis_v0.5.0: 1-layer BiLSTM trained from scratch (300-d embeddings, 50k vocabulary, 256 hidden units per direction, dropout 0.2) with early stopping, which ended training after epoch 9 and kept the lowest-validation-loss epoch-7 checkpoint; then its sigmoid head refit on the frozen encoder (unweighted BCE, lr 1e-3); max_length 512 | 0.2410 | 0.1179 | 0.3892 | 0.3640 | 0.0545 |
| 9 | transformer | baseline | Transformer encoder (RoBERTa family) | release cordis_v0.5.0: xlm-roberta-base fine-tuned for 3 epochs (lr 2e-5), then its sigmoid head refit on the frozen encoder for up to 20 epochs (unweighted BCE, lr 1e-3); max_length 512 | 0.2034 | 0.0861 | 0.3840 | 0.3135 | 0.0181 |
| - | hard_closed | new | NALTRA hierarchy-closed hard vote | k-of-M vote over ancestor-closed 586-node label sets, k tuned on val-B | - | - | 0.5596 | - | - |
| - | leave_one_out(-svm) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.2926 | - | - | - | - |
| - | leave_one_out(-hybrid_knn) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.3069 | - | - | - | - |
| - | leave_one_out(-bilstm) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.3954 | - | - | - | - |
| - | leave_one_out(-transformer) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.3936 | - | - | - | - |

### tr_test_noisy

| rank | system | role | architecture | model | micro_f1 | macro_f1 | hmicro_f1 | p_at_1 | top1_ece |
|---|---|---|---|---|---|---|---|---|---|
| 1 | svm | baseline | linear SVM over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k features, min_df 2, sublinear tf) + one LinearSVC (C 1.0) per label, 3-fold project-grouped Platt calibration | 0.4323 | 0.3682 | 0.5613 | 0.6313 | 0.1212 |
| 2 | weighted_soft | new | NALTRA weighted soft vote | raw member scores weighted by val-A micro-F1, global threshold tuned on val-B | 0.4211 | 0.3315 | 0.5436 | 0.6362 | 0.2719 |
| 3 | soft | new | NALTRA soft vote | mean of raw member scores, global threshold tuned on val-B | 0.4130 | 0.3198 | 0.5427 | 0.6152 | 0.2955 |
| 4 | hard_k | new | NALTRA hard vote | k-of-M vote of member label sets, k tuned on val-B for micro-F1 | 0.3809 | 0.2732 | 0.5183 | - | - |
| 5 | hybrid_knn | new | sparse + dense retrieval kNN with reciprocal rank fusion (Cormack et al. 2009, k = 60) | TF-IDF 1-2-grams (50k) cosine + intfloat/multilingual-e5-base mean-pooled embeddings (max_length 256), 50 neighbours per retriever, RRF k 60, fused neighbour label vote | 0.3673 | 0.2730 | 0.5173 | 0.5432 | 0.2397 |
| 6 | hard_majority | new | NALTRA hard vote | strict majority (more than M/2) of member label sets, member thresholds tuned on val-A | 0.2540 | 0.1246 | 0.3603 | - | - |
| 7 | naive_bayes | baseline | multinomial Naive Bayes over TF-IDF, one-vs-rest | TF-IDF word 1-2-grams (50k features, min_df 2, sublinear tf) + one binary MultinomialNB per label, alpha 0.03 | 0.2502 | 0.1311 | 0.3850 | 0.4065 | 0.1409 |
| 8 | bilstm | baseline | bidirectional LSTM | release cordis_v0.5.0: 1-layer BiLSTM trained from scratch (300-d embeddings, 50k vocabulary, 256 hidden units per direction, dropout 0.2) with early stopping, which ended training after epoch 9 and kept the lowest-validation-loss epoch-7 checkpoint; then its sigmoid head refit on the frozen encoder (unweighted BCE, lr 1e-3); max_length 512 | 0.2186 | 0.1054 | 0.3631 | 0.3458 | 0.0566 |
| 9 | transformer | baseline | Transformer encoder (RoBERTa family) | release cordis_v0.5.0: xlm-roberta-base fine-tuned for 3 epochs (lr 2e-5), then its sigmoid head refit on the frozen encoder for up to 20 epochs (unweighted BCE, lr 1e-3); max_length 512 | 0.1948 | 0.0793 | 0.3735 | 0.2953 | 0.0310 |
| - | hard_closed | new | NALTRA hierarchy-closed hard vote | k-of-M vote over ancestor-closed 586-node label sets, k tuned on val-B | - | - | 0.5412 | - | - |
| - | leave_one_out(-svm) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.2715 | - | - | - | - |
| - | leave_one_out(-hybrid_knn) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.2766 | - | - | - | - |
| - | leave_one_out(-bilstm) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.3763 | - | - | - | - |
| - | leave_one_out(-transformer) | new | NALTRA hard vote (ablation) | strict majority of the M-1 members left after dropping each member in turn | 0.3693 | - | - | - | - |

### Label distribution

all_negative_accuracy is the per-label accuracy of predicting no label at all.

| statistic | tuning | en_test |
|---|---|---|
| n_records | 9422 | 4711 |
| n_labels | 473 | 473 |
| mean_labels_per_record | 3.2314 | 3.2161 |
| median_labels_per_record | 3.0000 | 3.0000 |
| min_prevalence | 0.0013 | 0.0013 |
| median_prevalence | 0.0040 | 0.0040 |
| max_prevalence | 0.0563 | 0.0565 |
| rare_labels | 386 | 386 |
| rare_label_share | 0.8161 | 0.8161 |
| top_decile_positive_share | 0.3628 | 0.3637 |
| prevalence_ratio | 44.1667 | 44.3333 |
| zero_positive_labels | 0 | 0 |
| all_negative_accuracy | 0.9932 | 0.9932 |

## en_test

| system | micro_f1 | macro_f1 | hmicro_f1 | p_at_1 |
|---|---|---|---|---|
| svm | 0.5584 | 0.4893 | 0.6683 | 0.7455 |
| hybrid_knn | 0.3929 | 0.3121 | 0.5446 | 0.5672 |
| bilstm | 0.2891 | 0.1610 | 0.4606 | 0.4356 |
| transformer | 0.2476 | 0.1194 | 0.4382 | 0.3823 |
| naive_bayes | 0.2777 | 0.1976 | 0.4397 | 0.4549 |
| hard_majority | 0.3699 | 0.2125 | 0.4977 | - |
| hard_k | 0.4385 | 0.3451 | 0.5836 | - |
| hard_closed | - | - | 0.5960 | - |
| soft | 0.4975 | 0.4222 | 0.6279 | 0.6790 |
| weighted_soft | 0.5228 | 0.4426 | 0.6408 | 0.7077 |

## tr_test

| system | micro_f1 | macro_f1 | hmicro_f1 | p_at_1 |
|---|---|---|---|---|
| svm | 0.4886 | 0.4228 | 0.6152 | 0.6899 |
| hybrid_knn | 0.3817 | 0.2946 | 0.5338 | 0.5576 |
| bilstm | 0.2708 | 0.1544 | 0.4428 | 0.3991 |
| transformer | 0.2398 | 0.1113 | 0.4278 | 0.3632 |
| naive_bayes | 0.2753 | 0.1656 | 0.4327 | 0.4315 |
| hard_majority | 0.3399 | 0.1895 | 0.4688 | - |
| hard_k | 0.4154 | 0.3149 | 0.5631 | - |
| hard_closed | - | - | 0.5774 | - |
| soft | 0.4553 | 0.3709 | 0.5940 | 0.6396 |
| weighted_soft | 0.4723 | 0.3890 | 0.6004 | 0.6678 |

## cs_chunk_test

| system | micro_f1 | macro_f1 | hmicro_f1 | p_at_1 |
|---|---|---|---|---|
| svm | 0.5124 | 0.4551 | 0.6384 | 0.7124 |
| hybrid_knn | 0.3847 | 0.3023 | 0.5353 | 0.5680 |
| bilstm | 0.2818 | 0.1614 | 0.4545 | 0.4222 |
| transformer | 0.2467 | 0.1193 | 0.4379 | 0.3681 |
| naive_bayes | 0.2788 | 0.1942 | 0.4414 | 0.4551 |
| hard_majority | 0.3577 | 0.2052 | 0.4901 | - |
| hard_k | 0.4277 | 0.3367 | 0.5748 | - |
| hard_closed | - | - | 0.5888 | - |
| soft | 0.4807 | 0.4167 | 0.6155 | 0.6646 |
| weighted_soft | 0.5008 | 0.4329 | 0.6270 | 0.6863 |

## cs_sentence_test

| system | micro_f1 | macro_f1 | hmicro_f1 | p_at_1 |
|---|---|---|---|---|
| svm | 0.5080 | 0.4507 | 0.6338 | 0.7069 |
| hybrid_knn | 0.3834 | 0.2999 | 0.5354 | 0.5634 |
| bilstm | 0.2836 | 0.1633 | 0.4563 | 0.4267 |
| transformer | 0.2445 | 0.1182 | 0.4356 | 0.3662 |
| naive_bayes | 0.2800 | 0.1987 | 0.4414 | 0.4528 |
| hard_majority | 0.3585 | 0.2096 | 0.4932 | - |
| hard_k | 0.4243 | 0.3312 | 0.5719 | - |
| hard_closed | - | - | 0.5850 | - |
| soft | 0.4816 | 0.4161 | 0.6159 | 0.6589 |
| weighted_soft | 0.5029 | 0.4305 | 0.6284 | 0.6846 |

## en_test_noisy

| system | micro_f1 | macro_f1 | hmicro_f1 | p_at_1 |
|---|---|---|---|---|
| svm | 0.4801 | 0.4074 | 0.5951 | 0.6837 |
| hybrid_knn | 0.3740 | 0.2799 | 0.5268 | 0.5511 |
| bilstm | 0.2410 | 0.1179 | 0.3892 | 0.3640 |
| transformer | 0.2034 | 0.0861 | 0.3840 | 0.3135 |
| naive_bayes | 0.2617 | 0.1568 | 0.4023 | 0.4341 |
| hard_majority | 0.2838 | 0.1447 | 0.3934 | - |
| hard_k | 0.3993 | 0.2900 | 0.5383 | - |
| hard_closed | - | - | 0.5596 | - |
| soft | 0.4447 | 0.3470 | 0.5712 | 0.6500 |
| weighted_soft | 0.4586 | 0.3657 | 0.5758 | 0.6733 |

## tr_test_noisy

| system | micro_f1 | macro_f1 | hmicro_f1 | p_at_1 |
|---|---|---|---|---|
| svm | 0.4323 | 0.3682 | 0.5613 | 0.6313 |
| hybrid_knn | 0.3673 | 0.2730 | 0.5173 | 0.5432 |
| bilstm | 0.2186 | 0.1054 | 0.3631 | 0.3458 |
| transformer | 0.1948 | 0.0793 | 0.3735 | 0.2953 |
| naive_bayes | 0.2502 | 0.1311 | 0.3850 | 0.4065 |
| hard_majority | 0.2540 | 0.1246 | 0.3603 | - |
| hard_k | 0.3809 | 0.2732 | 0.5183 | - |
| hard_closed | - | - | 0.5412 | - |
| soft | 0.4130 | 0.3198 | 0.5427 | 0.6152 |
| weighted_soft | 0.4211 | 0.3315 | 0.5436 | 0.6362 |

## Significance

Paired project bootstrap, 10000 resamples, 95% percentile intervals, Holm-adjusted p. Best member on val-A: svm.

| test set | ensemble vs best member | micro-F1 diff [CI] | p (Holm) |
|---|---|---|---|
| en_test | hard_majority | -0.1884 [-0.1960, -0.1810] | 0.0007999 |
| en_test | hard_k | -0.1199 [-0.1256, -0.1142] | 0.0007999 |
| en_test | soft | -0.0609 [-0.0659, -0.0558] | 0.0007999 |
| en_test | weighted_soft | -0.0355 [-0.0401, -0.0310] | 0.0007999 |
| tr_test | hard_majority | -0.1487 [-0.1559, -0.1415] | 0.0007999 |
| tr_test | hard_k | -0.0732 [-0.0788, -0.0675] | 0.0007999 |
| tr_test | soft | -0.0333 [-0.0384, -0.0283] | 0.0007999 |
| tr_test | weighted_soft | -0.0163 [-0.0208, -0.0119] | 0.0007999 |
| cs_chunk_test | hard_majority | -0.1547 [-0.1621, -0.1474] | 0.0007999 |
| cs_chunk_test | hard_k | -0.0847 [-0.0902, -0.0791] | 0.0007999 |
| cs_chunk_test | soft | -0.0317 [-0.0365, -0.0270] | 0.0007999 |
| cs_chunk_test | weighted_soft | -0.0116 [-0.0159, -0.0073] | 0.0007999 |
| cs_sentence_test | hard_majority | -0.1494 [-0.1567, -0.1422] | 0.0007999 |
| cs_sentence_test | hard_k | -0.0836 [-0.0892, -0.0782] | 0.0007999 |
| cs_sentence_test | soft | -0.0264 [-0.0311, -0.0217] | 0.0007999 |
| cs_sentence_test | weighted_soft | -0.0050 [-0.0092, -0.0009] | 0.0174 |
| en_test_noisy | hard_majority | -0.1963 [-0.2040, -0.1884] | 0.0007999 |
| en_test_noisy | hard_k | -0.0808 [-0.0870, -0.0746] | 0.0007999 |
| en_test_noisy | soft | -0.0354 [-0.0406, -0.0301] | 0.0007999 |
| en_test_noisy | weighted_soft | -0.0215 [-0.0261, -0.0170] | 0.0007999 |
| tr_test_noisy | hard_majority | -0.1784 [-0.1860, -0.1708] | 0.0007999 |
| tr_test_noisy | hard_k | -0.0514 [-0.0572, -0.0454] | 0.0007999 |
| tr_test_noisy | soft | -0.0193 [-0.0244, -0.0142] | 0.0007999 |
| tr_test_noisy | weighted_soft | -0.0112 [-0.0157, -0.0067] | 0.0007999 |

| system | comparison | micro-F1 diff [CI] | p (Holm) |
|---|---|---|---|
| svm | en_test->tr_test | +0.0698 [+0.0642, +0.0754] | 0.0009999 |
| svm | en_test->cs_chunk_test | +0.0460 [+0.0416, +0.0503] | 0.0009999 |
| svm | en_test->cs_sentence_test | +0.0504 [+0.0458, +0.0548] | 0.0009999 |
| svm | en_test->en_test_noisy | +0.0783 [+0.0723, +0.0843] | 0.0009999 |
| svm | en_test->tr_test_noisy | +0.1260 [+0.1192, +0.1328] | 0.0009999 |
| hybrid_knn | en_test->tr_test | +0.0112 [+0.0068, +0.0157] | 0.0009999 |
| hybrid_knn | en_test->cs_chunk_test | +0.0083 [+0.0042, +0.0124] | 0.0009999 |
| hybrid_knn | en_test->cs_sentence_test | +0.0095 [+0.0054, +0.0135] | 0.0009999 |
| hybrid_knn | en_test->en_test_noisy | +0.0189 [+0.0148, +0.0233] | 0.0009999 |
| hybrid_knn | en_test->tr_test_noisy | +0.0257 [+0.0207, +0.0306] | 0.0009999 |
| bilstm | en_test->tr_test | +0.0183 [+0.0123, +0.0244] | 0.0009999 |
| bilstm | en_test->cs_chunk_test | +0.0073 [+0.0024, +0.0121] | 0.007599 |
| bilstm | en_test->cs_sentence_test | +0.0054 [+0.0008, +0.0103] | 0.0236 |
| bilstm | en_test->en_test_noisy | +0.0481 [+0.0425, +0.0536] | 0.0009999 |
| bilstm | en_test->tr_test_noisy | +0.0705 [+0.0634, +0.0773] | 0.0009999 |
| transformer | en_test->tr_test | +0.0079 [+0.0030, +0.0127] | 0.005399 |
| transformer | en_test->cs_chunk_test | +0.0009 [-0.0031, +0.0049] | 0.6493 |
| transformer | en_test->cs_sentence_test | +0.0032 [-0.0008, +0.0071] | 0.2384 |
| transformer | en_test->en_test_noisy | +0.0442 [+0.0390, +0.0495] | 0.0009999 |
| transformer | en_test->tr_test_noisy | +0.0528 [+0.0469, +0.0589] | 0.0009999 |
| naive_bayes | en_test->tr_test | +0.0024 [-0.0018, +0.0067] | 0.5415 |
| naive_bayes | en_test->cs_chunk_test | -0.0011 [-0.0039, +0.0017] | 0.5415 |
| naive_bayes | en_test->cs_sentence_test | -0.0023 [-0.0051, +0.0004] | 0.2988 |
| naive_bayes | en_test->en_test_noisy | +0.0159 [+0.0110, +0.0210] | 0.0009999 |
| naive_bayes | en_test->tr_test_noisy | +0.0275 [+0.0212, +0.0339] | 0.0009999 |
| hard_majority | en_test->tr_test | +0.0300 [+0.0239, +0.0360] | 0.0009999 |
| hard_majority | en_test->cs_chunk_test | +0.0123 [+0.0072, +0.0173] | 0.0009999 |
| hard_majority | en_test->cs_sentence_test | +0.0114 [+0.0065, +0.0164] | 0.0009999 |
| hard_majority | en_test->en_test_noisy | +0.0862 [+0.0797, +0.0927] | 0.0009999 |
| hard_majority | en_test->tr_test_noisy | +0.1160 [+0.1087, +0.1235] | 0.0009999 |
| hard_k | en_test->tr_test | +0.0231 [+0.0182, +0.0280] | 0.0009999 |
| hard_k | en_test->cs_chunk_test | +0.0108 [+0.0065, +0.0151] | 0.0009999 |
| hard_k | en_test->cs_sentence_test | +0.0141 [+0.0099, +0.0183] | 0.0009999 |
| hard_k | en_test->en_test_noisy | +0.0392 [+0.0338, +0.0447] | 0.0009999 |
| hard_k | en_test->tr_test_noisy | +0.0575 [+0.0516, +0.0636] | 0.0009999 |
| soft | en_test->tr_test | +0.0422 [+0.0374, +0.0471] | 0.0009999 |
| soft | en_test->cs_chunk_test | +0.0168 [+0.0128, +0.0207] | 0.0009999 |
| soft | en_test->cs_sentence_test | +0.0159 [+0.0118, +0.0199] | 0.0009999 |
| soft | en_test->en_test_noisy | +0.0528 [+0.0472, +0.0584] | 0.0009999 |
| soft | en_test->tr_test_noisy | +0.0844 [+0.0779, +0.0909] | 0.0009999 |
| weighted_soft | en_test->tr_test | +0.0505 [+0.0454, +0.0558] | 0.0009999 |
| weighted_soft | en_test->cs_chunk_test | +0.0221 [+0.0179, +0.0263] | 0.0009999 |
| weighted_soft | en_test->cs_sentence_test | +0.0199 [+0.0157, +0.0242] | 0.0009999 |
| weighted_soft | en_test->en_test_noisy | +0.0643 [+0.0584, +0.0702] | 0.0009999 |
| weighted_soft | en_test->tr_test_noisy | +0.1017 [+0.0949, +0.1084] | 0.0009999 |
