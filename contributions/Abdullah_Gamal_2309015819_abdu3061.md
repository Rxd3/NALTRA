# Student Contribution Report

**Student Name:** Abdullah 
**Student id:**2309015819 
**Role:** Evaluation & Benchmarking Lead  
**Project:** NALTRA (EuroSciVoc Multi-Label Classification)  

---

## Executive Summary
As the Evaluation & Benchmarking Lead for the NALTRA project, my primary responsibility was designing and implementing the end-to-end evaluation engine, metric suite, and performance benchmarking scripts. I established the mathematical and empirical standards used to evaluate all classification models against the 586-node EuroSciVoc taxonomy across clean, noisy, and cross-lingual text datasets.

---

## Technical Deliverables & Key Contributions
### 1. Core Evaluation Engine (`src/naltra/evaluation/`)
* **`metrics.py` (Metric Framework & Sparsity Resolution)**:
  * Researched and implemented global **Micro-F1**, **Macro-F1**, and standard accuracy metrics for extreme multi-label classification across 473 leaf categories.
  * Identified and documented the **99.3% accuracy trap** caused by label sparsity (where predicting all zero labels yields 99.3% standard accuracy) and justified the selection of Micro-F1 as the primary global evaluation metric.
* **`hierarchical.py` (Hierarchical Closure Metrics)**:
  * Designed hierarchical evaluation logic leveraging the 7-level EuroSciVoc tree structure (586 total nodes).
  * Implemented partial-credit scoring algorithms so models are fairly rewarded when predicting higher-level parent topics along the correct taxonomic branch (e.g., *Optics* instead of *Nonlinear Optics*).

### 2. Multi-Slice & Speed Benchmarking Pipeline (`scripts/`)
* **`scripts/evaluate_all.py` (Multi-Slice Evaluation)**:
  * Engineered automated multi-slice evaluation scripts to stress-test models across Clean English, Turkish, Code-Switched, and Noisy Text datasets.
* **`scripts/run_benchmark.py` (CPU Latency & Throughput Testing)**:
  * Created standardized latency benchmarks using 4 PyTorch CPU threads across 200 document batches.
  * Benchmarked inference speeds across classical baselines, neural architectures, and transformer models.

### 3. Baseline Validation & Empirical Findings
* **Calibrated Linear SVM (TF-IDF) Baseline**:
  * Established the calibrated SVM as our primary benchmark model.
  * Proved through empirical testing that Linear SVM achieved the top multi-label accuracy score (**0.558 Micro-F1** on clean English) while maintaining an ultra-fast production speed (**2.0 ms median CPU latency**).
  * Evaluated and documented the trade-offs of heavier neural baselines (BiLSTM), cross-lingual transformers (XLM-R), and zero-shot LLMs.

### 4. Testing, Code Quality & Verification
* Developed and maintained rigorous unit and integration tests for all evaluation scripts and pipeline modules, contributing directly to the repository's overall suite of **808 passing tests**.