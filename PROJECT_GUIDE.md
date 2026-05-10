# Project Guide

This file explains how to move around the project, where to add new code, and how to run experiments.

## Main Workflow

1. Add reusable code to the proper package, for example `models/`, `training/`, `evaluation/`, `data/`, `hpo/`, or `nas/`.
2. Add experiment settings to `experiments/*.yaml`.
3. Add or reuse a script in `scripts/` to run the experiment.
4. Use notebooks only as lightweight launchers and result viewers.
5. Save generated CSV/JSON/PNG outputs in `results/` or `plots/`.
6. Do not commit datasets, virtual environments, or large checkpoints.

## Directory Map

`data/`

Dataset code. Put dataloaders, train/validation/test split logic, and transforms here.

Current files:
- `data/dataloader.py` creates CIFAR-10 datasets and dataloaders.
- `data/transforms.py` defines CIFAR-10 normalization and augmentation.

`models/`

Model definitions. Add new neural networks here.

Examples:
- `models/baseline_cnn.py`
- `models/search_cnn.py`
- future `models/darts_model.py`

`training/`

Training utilities. Put reusable training loops, checkpoint logic, early stopping, and logging here.

Current file:
- `training/trainer.py`

`evaluation/`

Evaluation utilities. Put metrics, latency measurement, parameter counting, and test-set evaluation here.

Current file:
- `evaluation/metrics.py`

`experiments/`

YAML configuration files. These should describe experiment settings, not contain Python logic.

Examples:
- `experiments/baseline_cnn.yaml`
- `experiments/hpo_baseline.yaml`
- `experiments/evolutionary_nas.yaml`

`scripts/`

Executable experiment entrypoints. These scripts read configs, run training/search/evaluation, and save outputs.

Current files:
- `scripts/run_baseline.py`
- `scripts/run_hpo.py`
- `scripts/run_evolutionary_nas.py`

Run it with:

```bash
python scripts/run_baseline.py --config experiments/baseline_cnn.yaml
```

Run HPO with:

```bash
python scripts/run_hpo.py --config experiments/hpo_baseline.yaml
```

Run evolutionary NAS with:

```bash
python scripts/run_evolutionary_nas.py --config experiments/evolutionary_nas.yaml
```

`notebooks/`

Colab/Jupyter notebooks. Keep notebooks small. They should launch scripts, display tables/plots, and help with interactive inspection.

Current files:
- `notebooks/baseline_cnn_colab.ipynb`
- `notebooks/hpo_baseline_colab.ipynb`
- `notebooks/evolutionary_nas_colab.ipynb`
- `notebooks/colab_setup.md`

`utils/`

Small shared helpers used across the project.

Current files:
- `utils/config.py`
- `utils/plotting.py`
- `utils/reproducibility.py`

`results/`

Generated experiment outputs that are useful to commit, such as:
- CSV training logs
- JSON summaries
- small result tables

`plots/`

Generated plots that are useful for reports and analysis.

`checkpoints/`

Model weights. These are ignored by Git by default because they can be large.

`data/raw/`

Downloaded datasets. This is ignored by Git.

## Running Locally

Install dependencies with `uv`:

```bash
uv sync
```

Run the baseline experiment:

```bash
uv run python scripts/run_baseline.py --config experiments/baseline_cnn.yaml
```

Outputs are written to paths defined in the YAML config:

```text
results/baseline_training_log.csv
results/baseline_summary.json
results/baseline_training_curves.png
checkpoints/baseline_cnn.pt
```

Run the HPO baseline search:

```bash
uv run python scripts/run_hpo.py --config experiments/hpo_baseline.yaml
```

HPO outputs include:

```text
results/hpo_trials.csv
results/hpo_best_params.json
results/hpo_summary.json
results/hpo_best_training_log.csv
plots/hpo_optimization_history.png
plots/hpo_best_training_curves.png
checkpoints/hpo_best_baseline_cnn.pt
```

Run evolutionary NAS:

```bash
uv run python scripts/run_evolutionary_nas.py --config experiments/evolutionary_nas.yaml
```

Evolutionary NAS outputs include:

```text
results/evolutionary_population.csv
results/evolutionary_best_genome.json
results/evolutionary_summary.json
results/evolutionary_best_training_log.csv
plots/evolutionary_progress.png
plots/evolutionary_best_training_curves.png
checkpoints/evolutionary_best_cnn.pt
```

## Running In Colab

1. Push your code to GitHub.
2. Open `notebooks/baseline_cnn_colab.ipynb` from GitHub in Colab.
3. Set `Runtime > Change runtime type > GPU`.
4. Run all cells.
5. Download useful outputs from `results/`.
6. Commit downloaded result files from your local machine.

## What To Commit

Commit:
- source code
- notebooks
- YAML configs
- small CSV logs
- JSON summaries
- plots used in reports

Do not commit:
- `.venv/`
- `data/raw/`
- `checkpoints/*.pt`
- large downloaded/generated files

## Adding A New Model

1. Add the model in `models/new_model.py`.
2. Add model parameters to a YAML config in `experiments/`.
3. Add or update a script in `scripts/`.
4. Run the experiment.
5. Save CSV/JSON/PNG outputs.

## Adding A New Experiment

Create a new YAML file:

```text
experiments/my_experiment.yaml
```

Then run it with an existing or new script:

```bash
uv run python scripts/run_baseline.py --config experiments/my_experiment.yaml
```

If the experiment needs new behavior, add a new script in `scripts/`.

## Recommended Rule

Keep notebooks thin. If code is important enough to reuse or explain in the report, it probably belongs in a Python file, not only inside a notebook.
