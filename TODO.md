# TODO — Neural Architecture Search & Hyperparameter Optimization Research Project

## Project Goal

Build a research-oriented framework for:
- Hyperparameter Optimization (HPO)
- Evolutionary Neural Architecture Search (NAS)
- Hardware-Aware Optimization
- DARTS-inspired Differentiable Search

The project should optimize:
- Accuracy
- Model size
- Inference latency

using CNN architectures trained on the CIFAR-10 dataset.

---

# PROJECT CHECKLIST
- [X] Stage 1 — Repository & Environment Setup
- [X] Stage 2 — CIFAR-10 Dataset Pipeline
- [X] Stage 3 — Manual CNN Baseline
- [X] Stage 4 — Training Pipeline
- [X] Stage 5 — Hyperparameter Optimization (HPO)
- [ ] Stage 6 — Evolutionary NAS
- [ ] Stage 7 — Hardware-Aware Optimization
- [ ] Stage 8 — Pareto Optimization
- [ ] Stage 9 — DARTS-Inspired Differentiable Search
- [ ] Stage 10 — Experiment System
- [ ] Stage 11 — Logging & Results
- [ ] Stage 12 — Visualization & Analysis
- [ ] Stage 13 — Auto-Keras Comparison (Optional)
- [ ] Stage 14 — Final Report

---

# Stage 1 — Repository & Environment Setup

## Repository Structure

Create the following structure:

```text
project/
│
├── data/
├── models/
├── nas/
├── hpo/
├── training/
├── evaluation/
├── utils/
├── experiments/
├── notebooks/
├── plots/
├── reports/
└── results/
```

## Tasks

- [X] Initialize Git repository
- [X] Create project folder structure
- [X] Add `__init__.py` files to all Python modules
- [X] Configure Google Colab environment
- [X] Configure GPU runtime
- [X] Install dependencies
- [X] Create `requirements.txt`

## Required Libraries

- [X] PyTorch
- [X] Torchvision
- [X] Optuna
- [X] NumPy
- [X] Pandas
- [X] Matplotlib
- [X] PyYAML
- [X] tqdm

---

# Stage 2 — CIFAR-10 Dataset Pipeline

## Tasks

- [X] Download CIFAR-10 dataset
- [X] Create train/validation/test splits
- [X] Add normalization
- [X] Add data augmentation:
  - [X] RandomCrop
  - [X] RandomHorizontalFlip
  - [X] Normalize

## Deliverables

- [X] `data/dataloader.py`
- [X] `data/transforms.py`

---

# Stage 3 — Manual CNN Baseline

## Goal

Build a manually designed CNN architecture as the baseline model.

## Tasks

- [X] Implement baseline CNN model
- [X] Add configurable:
  - [X] Number of layers
  - [X] Filters
  - [X] Dropout
  - [X] Kernel sizes

## Metrics

Measure:
- [X] Accuracy
- [X] Loss
- [X] Parameter count
- [X] Training time
- [X] Inference latency

## Deliverables

- [X] `models/baseline_cnn.py`
- [X] `training/trainer.py`
- [X] `evaluation/metrics.py`

---

# Stage 4 — Training Pipeline

## Tasks

- [X] Implement generic training loop
- [X] Implement validation loop
- [X] Add early stopping
- [X] Add checkpoint saving
- [X] Add model loading
- [X] Add experiment logging

## Features

- [X] GPU support
- [X] Mixed precision training
- [X] Progress bars
- [X] TensorBoard or CSV logging

---

# Stage 5 — Hyperparameter Optimization (HPO)

## Goal

Automate tuning of CNN hyperparameters using Optuna.

## Search Space

Optimize:
- [X] Learning rate
- [X] Batch size
- [X] Optimizer type
- [X] Dropout
- [X] Number of filters
- [X] Number of layers

## Tasks

- [X] Integrate Optuna
- [X] Create objective function
- [X] Add pruning for weak trials
- [X] Save best hyperparameters
- [X] Compare HPO vs baseline

## Deliverables

- [X] `hpo/optuna_search.py`
- [X] HPO experiment notebook

---

# Stage 6 — Evolutionary NAS

## Goal

Implement an evolutionary algorithm for CNN architecture search.

## Search Space

Architecture parameters:
- [ ] Number of convolutional layers
- [ ] Filter sizes
- [ ] Kernel sizes
- [ ] Pooling type
- [ ] Skip connections
- [ ] Dropout

## Evolutionary Components

Implement:
- [ ] Population initialization
- [ ] Mutation
- [ ] Selection
- [ ] Crossover (optional)
- [ ] Aging / regularized evolution

## Tasks

- [ ] Create architecture genome representation
- [ ] Generate random architectures
- [ ] Train candidate models
- [ ] Evaluate candidate fitness
- [ ] Evolve population across generations

## Deliverables

- [ ] `nas/evolutionary_search.py`
- [ ] `nas/mutation.py`
- [ ] `nas/selection.py`

---

# Stage 7 — Hardware-Aware Optimization

## Goal

Optimize architectures using multiple objectives:
- Accuracy
- Latency
- Parameter count

## Tasks

- [ ] Measure inference latency
- [ ] Measure parameter count
- [ ] Add latency penalty to fitness
- [ ] Add parameter penalty to fitness

## Fitness Function

```math
Fitness = Accuracy - α * Params - β * Latency
```

## Deliverables

- [ ] `evaluation/latency.py`
- [ ] `nas/fitness.py`

---

# Stage 8 — Pareto Optimization

## Goal

Analyze trade-offs between:
- Accuracy
- Speed
- Model complexity

## Tasks

- [ ] Compute Pareto frontier
- [ ] Visualize Pareto-optimal models
- [ ] Compare efficient architectures

## Visualizations

- [ ] Accuracy vs Latency
- [ ] Accuracy vs Parameters
- [ ] Pareto Frontier

---

# Stage 9 — DARTS-Inspired Differentiable Search

## Goal

Implement simplified differentiable architecture search.

## Tasks

- [ ] Add operator weighting
- [ ] Implement softmax-weighted operations
- [ ] Train architecture parameters with gradient descent
- [ ] Compare with evolutionary NAS

## Candidate Operations

- [ ] Conv3x3
- [ ] Conv5x5
- [ ] Skip connection
- [ ] MaxPool
- [ ] AvgPool

## Deliverables

- [ ] `models/darts_model.py`
- [ ] `nas/darts_search.py`

---

# Stage 10 — Experiment System

## Goal

Create reusable experiment configuration system.

## Tasks

- [X] Add YAML experiment configs
- [X] Create experiment runner
- [X] Save experiment outputs automatically

## Example Config

```yaml
model:
  layers: 4
  filters: [32, 64, 128]

training:
  epochs: 10
  batch_size: 64

search:
  population_size: 20
```

## Deliverables

- [ ] `run_experiment.py`
- [X] `experiments/*.yaml`

---

# Stage 11 — Logging & Results

## Tasks

- [ ] Save metrics to CSV
- [ ] Save architectures
- [ ] Save plots
- [ ] Save checkpoints

## Result Format

```csv
architecture,accuracy,params,latency
cnn_v1,0.89,120000,5.2
cnn_v2,0.92,800000,14.7
```

---

# Stage 12 — Visualization & Analysis

## Tasks

- [ ] Plot training curves
- [ ] Plot HPO optimization history
- [ ] Plot NAS evolution progress
- [ ] Plot Pareto frontier
- [ ] Compare all approaches

## Final Analysis

Answer:
- Which method achieved the best accuracy?
- Which architecture was most efficient?
- How important is hardware-aware optimization?
- Is NAS worth the computational cost?

---

# Stage 13 — Auto-Keras Comparison (Optional)

## Goal

Compare custom NAS with Auto-Keras.

## Tasks

- [ ] Install Auto-Keras
- [ ] Run AutoML experiment
- [ ] Compare generated architectures
- [ ] Compare training efficiency

---

# Stage 14 — Final Report

## Report Sections

- [ ] Introduction
- [ ] Literature Review
- [ ] Methodology
- [ ] Experimental Setup
- [ ] Results
- [ ] Discussion
- [ ] Conclusion

## Include

- [ ] Tables
- [ ] Plots
- [ ] Pareto analysis
- [ ] Architecture comparisons
- [ ] Limitations
- [ ] Future work

---

# Recommended Development Order

## Phase 1
- Dataset pipeline
- Baseline CNN
- Training system

## Phase 2
- HPO with Optuna

## Phase 3
- Evolutionary NAS

## Phase 4
- Hardware-aware optimization

## Phase 5
- Pareto analysis

## Phase 6
- DARTS-inspired search

## Phase 7
- Final report and visualizations

---

# Most Important Engineering Rules

## DO NOT:
- Train every model for many epochs during search
- Use huge search spaces initially
- Put all logic inside notebooks

## ALWAYS:
- Log everything
- Save experiment results
- Use early stopping
- Keep experiments reproducible
- Separate code from notebooks

---

# Final Expected Outcome

A complete research-oriented AutoML framework capable of:
- Hyperparameter optimization
- Evolutionary NAS
- Hardware-aware architecture search
- Pareto-efficient model discovery

with full experimental analysis on CIFAR-10.
