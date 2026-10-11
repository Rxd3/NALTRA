# Presentation

`NALTRA_final.pptx` is the deck the team presents: five slides, followed by three hidden
backup slides for questions. The script has 209 conventional spoken words (about 236 with
initialisms spelled out). At 140 words per minute it takes about 1:41 before pauses and
slide changes. Check the two-minute limit with three timed rehearsals from five cues:
tree/task -> paired data -> methods/metric -> SVM and sample -> shifts, OOD, limits.

| Slide | Content |
| --- | --- |
| 1 | The task, hierarchical multi-label classification under language shift, shown on code-switched test project 895187 with its gold labels and added ancestors; the paired benchmark and flagging unfamiliar fields (the humanities held-out test) |
| 2 | The NALTRA CORDIS EN-TR benchmark, release 1.1.0: 31,407 paired projects, 283,002 aligned sentence pairs, 18,844 code-switched records, 473 labels in a 586-node tree and the 21,985 / 4,711 / 4,711 project split |
| 3 | Model names by role only: course baselines (Naive Bayes, SVM, BiLSTM, XLM-R) and new methods (hybrid kNN, NALTRA votes, zero-shot Kev, Jev, Laya); micro-F1 as the metric, since predicting no label scores 99.3% accuracy |
| 4 | Full-set micro-F1, where SVM leads (hierarchical 0.668 EN / 0.615 TR, 2.0 ms median CPU latency), and the 120-pair sample with Jev, Kev and Laya |
| 5 | EN -> TR drops, SVM's drop against every other trained member (Holm p 0.0008), near-OOD AUROC 0.927 EN / 0.924 TR, and the limits |
| B1 (hidden) | Micro-F1 lost from English, with 95% intervals, on TR, sentence code-switch, noisy EN and noisy TR, plus relative TR drops |
| B2 (hidden) | The 120-pair sample table with EN -> TR drops, Jev's pairwise intervals on English, and SVM's in-domain false-positive rate at 95% humanities recall |
| B3 (hidden) | Architectures and models: the role and the architecture -> concrete model behind each name on slide 3 |

The speaker notes of slides 1-5 hold the spoken script with its time stamps (0:00 to 1:49,
including two seconds per slide change); the backup notes say what questions each backup
answers. Every note names the files its numbers come from:

- [full-set summary](../../results/metrics/cordis_v1.1.0_ensemble/summary.json) and
  [drop-difference tests](../../results/metrics/cordis_v1.1.0_ensemble/robustness.json)
  under `results/metrics/cordis_v1.1.0_ensemble/`;
- [sample summary](../../results/metrics/cordis_v1.1.0_sample/summary.json) and
  [pairwise tests](../../results/metrics/cordis_v1.1.0_sample/pairwise.json) under
  `results/metrics/cordis_v1.1.0_sample/`;
- [latency](../../results/benchmarks/cordis_v1.1.0/benchmark_summary.json) and
  [near-OOD](../../results/ood/cordis_v1.1.0/summary.json) summaries;
- dataset counts from the [dataset guide](../dataset.md) and the
  [dataset card](../../data/README.md).

The [results guide](../results.md) explains these numbers and their limits.
