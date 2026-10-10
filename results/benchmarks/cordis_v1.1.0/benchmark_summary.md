| model | p50 ms | p95 ms | p99 ms | docs/s (batched) | size MB | parameters |
|---|---|---|---|---|---|---|
| naive_bayes | 1.77 | 2.39 | 2.91 | 812.6 | 96.5 | 23650473 |
| svm | 1.99 | 2.59 | 3.23 | 739.6 | 285.7 | 70954257 |
| hybrid_knn | 545.33 | 578.67 | 592.58 | 4.5 | 230.6 | 64288276 |
| bilstm | 13.41 | 17.62 | 18.99 | 77.2 | 66.6 | 16385433 |
| transformer | 376.40 | 453.13 | 538.67 | 2.5 | 1130.9 | 278407385 |

Runtime: torch 2.14.1+cu126, transformers 5.19.0, tokenizers 0.23.2, huggingface-hub 1.33.0, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1
