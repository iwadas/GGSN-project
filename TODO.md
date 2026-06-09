# TODO — Neural Architecture Search & Hyperparameter Optimization Research Project

## Project Goal

Build a research-oriented framework for:
- Hyperparameter Optimization (HPO)
- Evolutionary Neural Architecture Search (NAS)
- Hardware-Aware Optimization
- DARTS-inspired Differentiable Search

optimizing **Accuracy**, **Model size**, and **Inference latency** on CIFAR-10.

---

# Phase 1 — Dataset, Baseline CNN & Training Pipeline

**Status:** ✅ Done

## Deliverables
- [x] `data/dataloader.py` — CIFAR-10 download, train/val/test split
- [x] `data/transforms.py` — Normalizacja, RandomCrop, RandomHorizontalFlip, Cutout
- [x] `models/baseline_cnn.py` — Konfigurowalne CNN (layers, filtry, dropout, kernel sizes)
- [x] `training/trainer.py` — Trening, walidacja, early stopping, checkpointing, AMP
- [x] `evaluation/metrics.py` — Accuracy, loss, parameter count
- [x] `scripts/run_baseline.py` — Skrypt uruchomieniowy
- [x] `experiments/baseline_cnn.yaml` — Konfiguracja baseline'a
- [x] `notebooks/baseline_cnn_colab.ipynb` — Notebook Colab

## Wyniki
- Baseline: ~77% test accuracy, ~94K parametrów
- Krzywe uczenia: `plots/baseline_training_curves.png`
- Log: `results/baseline_training_log.csv`, `results/baseline_summary.json`

---

# Phase 2 — Hyperparameter Optimization (Optuna)

**Status:** ✅ Done

## Deliverables
- [x] `hpo/optuna_search.py` — Integracja Optuna, objective function, MedianPruner
- [x] Search space: learning rate, batch size, optimizer, dropout, filtry, layers, weight decay
- [x] Zapis najlepszych hiperparametrów i porównanie z baseline
- [x] `scripts/run_hpo.py`
- [x] `experiments/hpo_baseline.yaml`
- [x] `notebooks/hpo_baseline_colab.ipynb`

## Wyniki
- HPO best: ~85% test accuracy, ~6.21M parametrów (+8.09 pp nad baseline)
- Triale: `results/hpo_trials.csv`
- Best params: `results/hpo_best_params.json`
- Krzywe: `plots/hpo_best_training_curves.png`

---

# Phase 3 — Evolutionary NAS

**Status:** ✅ Done

## Deliverables
- [x] `nas/mutation.py` — Random genome, mutation (rate, grow/shrink), crossover
- [x] `nas/selection.py` — Elita, tournament selection
- [x] `nas/evolutionary_search.py` — Pętla ewolucyjna (populacja → selekcja → mutacja → aging)
- [x] `nas/fitness.py` — Accuracy fitness + hardware-aware fitness
- [x] `models/search_cnn.py` — CNN builder z genomu (skip connections, separable conv, dilation)
- [x] `scripts/run_evolutionary_nas.py`
- [x] `experiments/evolutionary_nas.yaml`
- [x] `notebooks/evolutionary_nas_colab.ipynb`

## Wyniki
- Evolutionary best: ~87% test accuracy, ~373K parametrów
- Populacja: `results/evolutionary_population.csv`
- Pareto frontier: `results/evolutionary_pareto_frontier.csv`
- Wykresy: progres ewolucji, Pareto frontier, Accuracy vs Latency/Params

---

# Phase 4 — Hardware-Aware & Pareto Optimization

**Status:** ✅ Done

## Deliverables
- [x] `evaluation/latency.py` — Pomiar inferencji (warmup, CUDA sync)
- [x] `evaluation/pareto.py` — Pareto frontier + wykresy (Accuracy vs Latency, Accuracy vs Params, Pareto Frontier)
- [x] `nas/fitness.py` — Fitness = Accuracy - α·Params - β·Latency
- [x] Wykresy: `plots/evolutionary_accuracy_vs_latency.png`, `plots/evolutionary_accuracy_vs_parameters.png`, `plots/evolutionary_pareto_frontier.png`

---

# Phase 5 — DARTS-Inspired Differentiable Search

**Status:** ✅ Done

## Deliverables
- [x] `models/darts_model.py` — MixedOp z learnable alpha, softmax-weightowane operacje
- [x] Candidate operations: Conv3x3, Conv5x5, Skip connection, MaxPool, AvgPool
- [x] `nas/darts_search.py` — Bi-level optimization, temperature annealing, entropy regularization
- [x] Derive discrete architecture z wytrenowanych alpha
- [x] Porównanie z Evolutionary NAS
- [x] `scripts/run_darts_search.py`
- [x] `experiments/darts_search.yaml`
- [x] `notebooks/darts_search_colab.ipynb`

## Wyniki
- DARTS best: ~81% test accuracy, ~128K parametrów, ~0.70ms latency
- Alpha log: `results/darts_alpha_log.csv`
- Derived genome: `results/darts_derived_genome.json`
- Wykresy: alpha convergence, training curves

---

# Phase 6 — Porównanie metod i raport końcowy

**Status:** ⬜ Do zrobienia

## Porównanie metod (1 notebook/scenariusz)
- [ ] Jeden wspólny wykres Accuracy vs Params dla wszystkich 4 metod
- [ ] Jeden wspólny wykres Accuracy vs Latency dla wszystkich 4 metod
- [ ] Tabela porównawcza (accuracy, params, latency, FLOPS)
- [ ] Która metoda wygrywa w której kategorii?

## Final Report (`reports/`)
- [ ] Wstęp i cel projektu
- [ ] Opis metod (Baseline, HPO, Evolutionary NAS, DARTS)
- [ ] Eksperymenty i wyniki
- [ ] Dyskusja (kompromisy, ograniczenia, wnioski)
- [ ] Future work

## Opcjonalnie — Auto-Keras Comparison
- [ ] Instalacja Auto-Keras
- [ ] Uruchomienie AutoML
- [ ] Porównanie architektur i wydajności

---

# Phase 7 — Ensemble metod

**Status:** ✅ Done

**Wyniki:**

| Model | Accuracy | Parametry |
|---|---|---|
| Baseline | 76.98% | 95K |
| HPO | 85.07% | 6.21M |
| Evolutionary NAS | **86.58%** | 373K |
| DARTS | 80.60% | 128K |
| **Ensemble (soft)** | **86.10%** | — |
| **Ensemble (hard)** | **84.67%** | — |

Soft voting nie bije najlepszej solo metody (Evo 86.58%) — ensemble jest zdominowany przez baseline (77%). Wariant bez baseline'a może dać lepszy wynik.

## Pliki wynikowe

| Plik | Opis |
|---|---|
| `results/ensemble_summary.json` | Accuracy każdej metody + ensemble (soft/hard) |
| `plots/ensemble_comparison.png` | Wykres słupkowy porównawczy |
| `scripts/run_ensemble.py` | Skrypt ensemble |
| `plots/confusion_matrix_{model}.png` | Macierz pomyłek dla każdego modelu + ensemble |

---

# Phase 8 — Knowledge Distillation

**Status:** ⬜ Do zrobienia

**Wymaganie wstępne:** Checkpoint HPO (`checkpoints/hpo_best_baseline_cnn.pt`) jako teacher oraz DARTS-derived genome (`results/darts_derived_genome.json`) do zbudowania studenta.

---

## Cel

Skompresować wiedzę z dużego modelu HPO (6.21M params, **teacher**, ~85% accuracy) do małego modelu DARTS-style (128K params, **student**) używając distillation loss. Pokazać, że student po dystylacji jest bliżej nauczyciela niż oryginalny DARTS.

## Koncepcja

Zamiast trenować studenta tylko z etykietami (hard targets), dokładamy **soft loss** — student uczy się naśladować rozkład prawdopodobieństw nauczyciela:

```
total_loss = α · KL_div(softmax(student/T), softmax(teacher/T)) + (1-α) · CrossEntropy(student, labels)
```

gdzie:
- `T` — temperatura (im wyższa, tym bardziej miękki rozkład)
- `α` — waga distillation loss (0.7 = głównie naśladuje nauczyciela)
- Student dostaje też prawdziwe etykiety (hard loss) — nie może uczyć się błędów nauczyciela

## Zadania

- [ ] **Implementacja w `training/distillation.py`**:
  - `distillation_loss(student_logits, teacher_logits, labels, temperature, alpha)`:
    - `soft_loss = KL_div(F.log_softmax(student/T), F.softmax(teacher/T)) * T²`
    - `hard_loss = CrossEntropy(student_logits, labels)`
    - `total_loss = alpha * soft_loss + (1-alpha) * hard_loss`
  - Funkcja pomocnicza do trenowania studenta z distillation

- [ ] **Skrypt `scripts/run_distillation.py`** który:
  - Ładuje **teacher**: `BaselineCNN` z `checkpoints/hpo_best_baseline_cnn.pt`
  - Buduje **student**: `SearchCNN` z genomu `results/darts_derived_genome.json`
  - Zamraża nauczyciela (`model.eval()`, `torch.no_grad()`)
  - **Prekomputowuje logity nauczyciela** na całym treningowym zbiorze (oszczędność czasu — teacher nie musi być odpalany przy każdej epoce)
  - Trenuje studenta z distillation loss przez 30 epok
  - Grid search hiperparametrów:
    - **Temperatura**: T = 1, 2, 4, 8
    - **Alpha**: α = 0.3, 0.5, 0.7
    - Razem: **12 wariantów**
  - Zapisuje wyniki dla każdej kombinacji

- [ ] **Ewaluacja i porównanie**:
  - Accuracy studenta (distilled) vs studenta (oryginalny DARTS)
  - Accuracy studenta vs nauczyciela (HPO)
  - Ile udało się odzyskać z luki accuracy: `(acc_distilled - acc_original_darts) / (acc_teacher - acc_original_darts) * 100%`
  - Kompresja: 6.21M → 128K parametrów (~98% redukcji)

- [ ] **Wykresy**:
  - `plots/distillation_training_curves.png` — krzywe treningu studenta (loss, accuracy)
  - `plots/distillation_comparison.png` — słupki: teacher vs student_original vs student_distilled (najlepszy T/α)
  - `plots/distillation_temperature_sweep.png` — accuracy od temperatury (dla ustalonego α)

## Spodziewany wynik

Luka do odzyskania: 85.07% (teacher) − 80.60% (student original) = **4.47 pp**

| Model | Accuracy | Parametry |
|---|---|---|
| Teacher (HPO) | **85.07%** | 6.21M |
| Student (oryginalny DARTS) | **80.60%** | 128K |
| **Student distilled** | **~82.8–83.7%** | **128K** |
| Recovery rate | 50–70% luki | — |

- Gain: ~2.2–3.1 pp nad oryginalnym DARTSem
- Strata do nauczyciela: ~1.5–2.5 pp przy **48× mniejszym modelu**
- Wyższa temperatura (T=4–8) powinna działać lepiej — bardziej miękki rozkład niesie więcej informacji o relacjach między klasami
- Czas: ~2h na T4 (12 wariantów × ~10 min)

## Pliki wynikowe

| Plik | Opis |
|---|---|
| `training/distillation.py` | Loss function + training helper |
| `scripts/run_distillation.py` | Skrypt |
| `results/distillation_summary.json` | Wyniki dla różnych T i α |
| `results/distillation_training_log.csv` | Per-epoch log najlepszego wariantu |
| `plots/distillation_training_curves.png` | Training curves |
| `plots/distillation_comparison.png` | Accuracy comparison |
| `plots/distillation_temperature_sweep.png` | Wpływ temperatury na accuracy |
| `checkpoints/distilled_student.pt` | Student checkpoint (najlepszy wariant) |

---

# Uwagi / notatki

- Logging i zapisywanie wyników (CSV, checkpointy, JSON) dzieje się automatycznie w każdej fazie — nie jest osobnym etapem.
- Wizualizacje (training curves, Pareto, alpha convergence) są generowane per-notebook. Phase 6 to dopiero wspólne zestawienie.
- **Checkpointy wag są w `.gitignore`** — trzeba je wygenerować lokalnie lub w Colab przed Phases 7-8. Skrypty z Phase 1-5 automatycznie tworzą checkpointy podczas final retrain.
- Phase 7 (ensemble) wymaga tylko inference na testsecie — **nie wymaga GPU** poza wczytaniem modeli.
- Phase 8 (distillation) wymaga trenowania studenta — **wskazany GPU** (szac. 15-20 min na T4 dla wszystkich wariantów T/α).
