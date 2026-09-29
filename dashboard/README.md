# NALTRA dashboard

This Streamlit app is the UI integration surface. It will provide text input, model selection, label confidence, hierarchy paths, OOD status, explanations, and side-by-side model comparison.

Start it from the repository root after installation:

```bash
make dashboard
```

The initial screen intentionally returns no fabricated predictions. Connect it to the shared `PredictionPipeline` only after at least one trained model is available.
