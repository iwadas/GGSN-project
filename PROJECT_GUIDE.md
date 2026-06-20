# Project Guide

This file explains how to move around the project, where to add new code, and how to run experiments.

For **detailed code-level documentation** (architecture, results, analysis), see [`DOCUMENTATION.md`](DOCUMENTATION.md).

## Why This Structure (and Not 5 Separate Notebooks)?

The project is split into **reusable Python modules** + **thin Colab notebooks** instead of having each experiment in a standalone notebook. This was a deliberate decision:

1. **DRY (Don't Repeat Yourself).** Data loading, training loops, evaluation metrics, and plotting are shared across all experiments. In a notebook-per-experiment approach, each notebook would duplicate the same 100+ lines of boilerplate. Here, shared code lives once in `training/`, `data/`, `evaluation/`, `utils/` and is imported everywhere.

2. **Colab-friendly.** Each notebook is a lightweight launcher — it clones the repo, installs dependencies, and calls a Python script. Notebooks stay small (no embedded 500-line training loops), open instantly in Colab, and are readable at a glance.

3. **Maintainability.** A bug in the training loop is fixed once in `training/trainer.py`, not hunted across 5 notebooks. Adding a new experiment means writing a script + YAML config, not copy-pasting a notebook and risking drift.

4. **Reproducibility.** Python scripts + YAML configs are deterministic, reviewable, and can be run from CLI without ever opening a notebook. Notebooks are used only for interactive inspection and result display.

5. **Where are the solutions?** Each experiment has a dedicated script (`scripts/run_*.py`) that performs the full pipeline (search → retrain → evaluate). The notebooks are entrypoints for Colab. The complete analysis and results are documented in [`DOCUMENTATION.md`](DOCUMENTATION.md) — refer there for architecture details, result tables, plots, and cross-method comparison.

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

Current files:
- `models/baseline_cnn.py` — configurable baseline CNN.
- `models/search_cnn.py` — searchable CNN used by evolutionary NAS / DARTS.
- `models/darts_model.py` — DARTS differentiable search model.

`training/`

Training utilities. Put reusable training loops, checkpoint logic, early stopping, logging, and distillation here.

Current files:
- `training/trainer.py` — training loop, checkpoints, early stopping.
- `training/distillation.py` — knowledge distillation (teacher–student).

`evaluation/`

Evaluation utilities. Put metrics, latency measurement, parameter counting, and Pareto analysis here.

Current files:
- `evaluation/metrics.py` — accuracy, parameter count, evaluation loop.
- `evaluation/latency.py` — inference latency measurement.
- `evaluation/pareto.py` — Pareto frontier computation and plotting.

`hpo/`

Hyperparameter optimization code.

Current files:
- `hpo/optuna_search.py` — Optuna study creation, objective, pruning, best-model retrain.

`nas/`

Neural Architecture Search code.

Current files:
- `nas/mutation.py` — genome representation, random generation, mutation.
- `nas/selection.py` — elite selection, tournament selection.
- `nas/fitness.py` — accuracy and hardware-aware fitness functions.
- `nas/evolutionary_search.py` — evolutionary search loop.
- `nas/darts_search.py` — DARTS differentiable search loop.

`experiments/`

YAML configuration files. These should describe experiment settings, not contain Python logic.

Current files:
- `experiments/baseline_cnn.yaml`
- `experiments/hpo_baseline.yaml`
- `experiments/evolutionary_nas.yaml`
- `experiments/darts_search.yaml`

`scripts/`

Executable experiment entrypoints. These scripts read configs, run training/search/evaluation, and save outputs.

Current files:
- `scripts/run_baseline.py`
- `scripts/run_hpo.py`
- `scripts/run_evolutionary_nas.py`
- `scripts/run_darts_search.py`
- `scripts/run_distillation.py`
- `scripts/run_ensemble.py`

Run any script with:

```bash
uv run python scripts/run_baseline.py --config experiments/baseline_cnn.yaml
```

`notebooks/`

Colab/Jupyter notebooks. Keep notebooks small. They should launch scripts, display tables/plots, and help with interactive inspection.

Current files:
- `notebooks/baseline_cnn_colab.ipynb`
- `notebooks/hpo_baseline_colab.ipynb`
- `notebooks/evolutionary_nas_colab.ipynb`
- `notebooks/darts_search_colab.ipynb`
- `notebooks/knowledge_distillation_colab.ipynb`
- `notebooks/ensemble_colab.ipynb`
- `notebooks/run_all_experiments.ipynb` — runs all 5 experiments sequentially.
- `notebooks/colab_setup.md`

`utils/`

Small shared helpers used across the project.

Current files:
- `utils/config.py` — YAML config loader.
- `utils/plotting.py` — training curves plotting.
- `utils/reproducibility.py` — deterministic seed setting.

`understanding/`

Background reading on methods used in the project.

Current files:
- `understanding/00_overview.md`
- `understanding/01_HPO.md`
- `understanding/02_evolutionary_nas.md`
- `understanding/03_darts.md`
- `understanding/04_hardware_aware_pareto.md`
- `understanding/05_ensemble_distillation.md`

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

Run DARTS search:

```bash
uv run python scripts/run_darts_search.py --config experiments/darts_search.yaml
```

DARTS outputs include:

```text
results/darts_alpha_log.csv
results/darts_derived_genome.json
results/darts_summary.json
results/darts_best_training_log.csv
plots/darts_alpha_convergence.png
plots/darts_best_training_curves.png
checkpoints/darts_best_cnn.pt
```

Run knowledge distillation:

```bash
uv run python scripts/run_distillation.py
```

Distillation outputs include:

```text
results/distillation_summary.json
plots/distillation_comparison.png
```

Run ensemble evaluation:

```bash
uv run python scripts/run_ensemble.py
```

Ensemble outputs include:

```text
results/ensemble_summary.json
plots/ensemble_confusion_matrix.png
```

## Running In Colab

1. Push your code to GitHub.
2. Open the relevant notebook from `notebooks/` on GitHub in Colab (see the table below).
3. Set `Runtime > Change runtime type > GPU`.
4. Run all cells.
5. Download useful outputs from `results/`.
6. Commit downloaded result files from your local machine.

| Experiment | Notebook |
|---|---|
| Baseline CNN | `notebooks/baseline_cnn_colab.ipynb` |
| HPO with Optuna | `notebooks/hpo_baseline_colab.ipynb` |
| Evolutionary NAS | `notebooks/evolutionary_nas_colab.ipynb` |
| DARTS Search | `notebooks/darts_search_colab.ipynb` |
| Knowledge Distillation | `notebooks/knowledge_distillation_colab.ipynb` |
| Ensemble | `notebooks/ensemble_colab.ipynb` |
| All experiments (sequential) | `notebooks/run_all_experiments.ipynb` |

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
