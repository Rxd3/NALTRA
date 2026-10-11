# Team ownership

Owners identify the primary reviewers for each area. Contributors can work across
areas; coordinate changes to shared interfaces before implementation.

| Contributor | Area | Main paths |
| --- | --- | --- |
| Almutesim Alakili | General support | Supports all areas |
| Rasit Isaoglu | Technical lead and core ML | Neural models, Jev/Laya, pipeline, schemas, configurations, training, direct evaluation |
| Mohamed Kamaleldin | UI and slides | `scripts/build_dashboard.py`, `src/naltra/reporting/`, `docs/results/` |
| Omar | Data and taxonomy | `data/`, `taxonomy/`, `src/naltra/data/`, dataset and taxonomy guides |
| Raouf | Classical ML and voting | Naive Bayes, SVM, their configurations, `src/naltra/pipeline/ensemble.py` |
| Abdullah Gamal | Evaluation and benchmarking | `src/naltra/evaluation/`, `results/`, evaluation guide, `scripts/run_benchmark.py` |

Raouf owns the ensemble implementation within the broader pipeline area coordinated
by the technical lead. Changes to the prediction contract, taxonomy IDs, split
definitions, or evaluation semantics need review across the affected areas.

Individual work is documented in the [contribution records](../contributions/).
See [Contributing](../CONTRIBUTING.md) for development and pull request instructions.
