# Stroke Signal Multimodal Fusion

Clean GitHub-ready code for the signal-based stroke classification pipeline.

The repository keeps only the pieces needed to run the main experiment flow:

1. Signal-only model.
2. Feature-only MLP model with architecture `128, 128, 64, 32, 1`.
3. Interaction early-fusion model.
4. Hybrid fusion using `hybrid_logit_equal`.

## Layout

```text
github_code/
  examples/
    run_dummy_pipeline.py
  scripts/
    train/
      train_signal_only.py
      train_feature_only.py
      train_interaction_early_fusion.py
    fusion/
      make_hybrid_fusion.py
  src/
    stroke_multimodal/
      models/
        interaction_early_fusion.py
```

## Setup

```bash
pip install -r requirements.txt
```

PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

Bash:

```bash
export PYTHONPATH="$PWD/src"
```

## Dummy End-to-End Run

This creates synthetic data and runs the full code path. The metrics are not meaningful; this checks that the pipeline, file formats, and output paths work.

```powershell
py examples\run_dummy_pipeline.py
```

Outputs are written to:

```text
outputs/dummy_pipeline/
```

## Real Data Pipeline

Run these scripts from the `github_code` folder. Override input and output paths with environment variables.

```powershell
py scripts\train\train_signal_only.py
py scripts\train\train_feature_only.py
py scripts\train\train_interaction_early_fusion.py
py scripts\fusion\make_hybrid_fusion.py
```

Default locations:

```text
data/signal_folds/
data/features/
data/multimodal_folds/
outputs/
```

Clinical data and generated outputs are intentionally ignored by git.
