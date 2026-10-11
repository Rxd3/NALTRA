#!/usr/bin/env bash
# Run the local installation without activating its virtual environment.
set -euo pipefail

NALTRA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NALTRA_PYTHON="$NALTRA_ROOT/.venv/bin/python"
NALTRA_ARTIFACTS="$NALTRA_ROOT/models/local_cpu_v1/cordis_h2020/en_tr_direct"

if [[ ! -x "$NALTRA_PYTHON" ]]; then
    echo "Missing .venv. Follow README.md to install NALTRA first." >&2
    exit 1
fi

cd "$NALTRA_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$NALTRA_ROOT/.venv/.cache/matplotlib}"
export HF_HOME="${HF_HOME:-$NALTRA_ROOT/.cache/huggingface}"

NALTRA_COMMAND="${1:-dashboard}"
if [[ $# -gt 0 ]]; then shift; fi

case "$NALTRA_COMMAND" in
    dashboard)
        if [[ $# -eq 0 ]] && curl --noproxy '*' --fail --silent --max-time 2 \
            http://127.0.0.1:8000/results/ | cmp -s docs/results/index.html -; then
            echo "NALTRA is already running: http://localhost:8000/results/"
            exit 0
        fi
        echo "NALTRA results: http://localhost:8000/results/ (Ctrl+C to stop)"
        exec "$NALTRA_PYTHON" -m http.server 8000 --bind 127.0.0.1 --directory docs "$@"
        ;;
    classify)
        exec "$NALTRA_PYTHON" scripts/classify.py --artifacts "$NALTRA_ARTIFACTS" --models svm "$@"
        ;;
    train)
        export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-4}"
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
        export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
        exec "$NALTRA_PYTHON" scripts/train_all.py --prebuilt-release \
            --models naive_bayes svm --device cpu --output-dir models/local_cpu_v1 "$@"
        ;;
    check)
        "$NALTRA_PYTHON" -m pip check
        "$NALTRA_PYTHON" scripts/validate_datasets.py --prebuilt-release
        exec "$NALTRA_PYTHON" -m pytest "$@"
        ;;
    help|--help|-h)
        echo 'Usage: ./run.sh [dashboard|classify|train|check|help]'
        echo 'Example: ./run.sh classify --text "This project develops laser sensors."'
        ;;
    *)
        echo "Unknown command: $NALTRA_COMMAND. Run ./run.sh help." >&2
        exit 2
        ;;
esac
