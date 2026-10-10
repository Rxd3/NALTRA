# Team Member Contribution Report

- **Name**: Abdullah Gamal
- **Student ID**: <YOUR_STUDENT_ID>
- **GitHub Username**: abdu3061
- **Role**: Evaluation & Benchmarking Lead

---

## What I Owned
I was responsible for the entire evaluation and benchmarking pipeline for our project. My main focus was making sure we could accurately measure how smart, fast, and reliable our models were across different languages and noisy text conditions.

Files and modules I created and maintained:
- `src/naltra/evaluation/` (metrics engine and hierarchical scoring)
- `scripts/evaluate_all.py` (automated testing across language/noise slices)
- `scripts/run_benchmark.py` (measuring speed, latency, and CPU memory)
- `results/` (storing JSON evaluation outputs)
- `docs/evaluation.md` (documentation for the evaluation setup)

---

## What I Did

1. **Built the Metrics Engine (`metrics.py` & `hierarchical.py`)**
   - Implemented standard evaluation metrics like Precision, Recall, Micro-F1, and Macro-F1.
   - Built the hierarchical scoring logic to give partial credit when predictions match ancestor nodes in our EuroSciVoc taxonomy tree.

2. **Created Automated Evaluation Scripts (`evaluate_all.py`)**
   - Wrote scripts that automatically run predictions through our metrics engine across all test sets: English, machine-translated Turkish, code-switched text, and noisy text.
   - Saved all structured results directly into `results/metrics/` so the team could compare baseline models against our main NALTRA model side-by-side.

3. **Benchmarked System Speed & Efficiency (`run_benchmark.py`)**
   - Measured real-world CPU speed for every model, capturing p50, p95, and p99 latency percentiles alongside throughput (documents per second).

4. **Testing & Team Integration**
   - Maintained full test coverage to make sure nothing broke, verifying all 116 unit tests passed.
   - Managed remote git branches and merged my work into `main` via Pull Request #2.