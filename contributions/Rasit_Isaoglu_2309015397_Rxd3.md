# Contribution: Raşit İsaoğlu

- **Name:** Raşit İsaoğlu
- **Student ID:** 2309015397
- **GitHub:** Rxd3
- **Role:** Technical lead and core ML member, with integration and repository cleanup support.

## What I built

**Project foundation and prediction pipeline**

- Initial repository structure, configurations, shared model interface, and prediction schema.
- Shared prediction pipeline with language detection, thresholds, hierarchy paths, and OOD handling.
- Initial hard, soft, and weighted-soft runtime voting, later extended by Raouf for the benchmark.

**Neural models**

- BiLSTM and Transformer training and inference, with early stopping, GPU support, and saved model artifacts.
- Frozen-encoder classifier-head refitting and validation-based threshold selection for the `cordis_v0.5.0` neural bundle.

**Laya and evaluation**

- Pinned local multilingual Laya integration and its evaluation workflow.
- Resumable evaluation using a prediction cache, so long runs can continue after interruptions.
- Extended the existing evaluator for neural models and code-switch tracks, with targeted regression tests.

**Integration support**

- Helped Omar's data work by strengthening release checks, imported sentence alignment validation, and code-switch audits.
- Helped extend Abdullah's evaluation work for neural models and frozen thresholds, and integrated the related team branches.

**Final review and documentation**

- Coordinated an audit and critical fixes for prediction validation, packaging, and benchmark inference settings.
- Simplified the README, reorganized documentation, removed temporary audit files, and documented the artifact handoff.

## Key results

The team benchmark reports these neural-model micro-F1 scores on the 4,711-project test sets (EN / TR):

- **BiLSTM:** 0.289 / 0.271.
- **Transformer:** 0.248 / 0.240.

The team now reports Laya on the 120-pair sample as a new method that is scored but does not vote (0.068 EN / 0.046 TR micro-F1). The dataset, classical models, full benchmark, and reporting retain their respective team contributors' credit.
