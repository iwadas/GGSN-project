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
- Baseline: ~68% test accuracy, ~94K parametrów
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
- HPO best: ~84% test accuracy, ~1.56M parametrów (+16.58 pp nad baseline)
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
- Evolutionary best: ~84% test accuracy, ~710K parametrów
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
- DARTS best: ~79% test accuracy, ~260K parametrów, ~0.53ms latency
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

**Status:** ⬜ Do zrobienia

**Wymaganie wstępne:** Checkpointy wag modeli (`checkpoints/*.pt`) nie istnieją lokalnie — są w `.gitignore` i były tylko w runtime Colaba. Przed ensemble trzeba je wygenerować:

| Checkpoint | Jak zdobyć | Szac. czas na T4 |
|---|---|---|
| `checkpoints/baseline_cnn.pt` | `uv run python scripts/run_baseline.py` | ~5 min |
| `checkpoints/hpo_best_baseline_cnn.pt` | `uv run python scripts/run_hpo.py` | ~15 min (50 epok final retrain) |
| `checkpoints/evolutionary_best_cnn.pt` | `uv run python scripts/run_evolutionary_nas.py` | ~10 min (25 epok) |
| `checkpoints/darts_best_cnn.pt` | `uv run python scripts/run_darts_search.py` | ~10 min (25 epok) |

**Razem: ~40 min na GPU (Colab T4).** Skrypty zrobią final retrain najlepszej architektury i zapiszą wagę do `checkpoints/`.

---

## Cel

Załadować wszystkie 4 wytrenowane modele i połączyć ich predykcje na testsecie — pokazać, że ensemble bije każdą metodę solo.

## Zadania

- [ ] **Przygotowanie — retrain 4 modeli** (jeśli checkpointy nie istnieją):
  - Kolejno odpalić skrypty z Phase 1-5 na GPU
  - Zweryfikować czy `checkpoints/` zawiera 4 pliki `.pt`

- [ ] **Skrypt `scripts/run_ensemble.py`** który:
  - Ładuje 4 checkpointy z `checkpoints/` i mapuje do odpowiednich klas:
    - `baseline_cnn.pt` → `BaselineCNN` (models/baseline_cnn.py)
    - `hpo_best_baseline_cnn.pt` → `BaselineCNN` (ten sam model, inne hiperparametry)
    - `evolutionary_best_cnn.pt` → `SearchCNN` (models/search_cnn.py — genom z `results/evolutionary_best_genome.json`)
    - `darts_best_cnn.pt` → `SearchCNN` (models/search_cnn.py — genom z `results/darts_derived_genome.json`)
  - Uruchamia inferencję na całym testsecie dla każdego modelu osobno
  - Implementuje 2 warianty ensemble:
    - **Soft voting** — `torch.stack([logits_1, logits_2, ...]).mean(0)` → argmax
    - **Hard voting** — większościowy wybór klasy po argmax każdego modelu
  - Liczy accuracy: każda metoda solo + ensemble soft + ensemble hard
  - Zapisuje wyniki do `results/ensemble_summary.json`

- [ ] **Wykres `plots/ensemble_comparison.png`**:
  - Słupki accuracy: baseline, HPO, evo, DARTS, ensemble (soft), ensemble (hard)
  - Linia lub adnotacja pokazująca improvement nad najlepszą solo metodą

- [ ] **Opcjonalnie: subset ensemble** — ensemble bez baseline'a, top-3, tylko top-2 itp.

## Spodziewany wynik

- Ensemble soft voting: ~85-86% test accuracy (vs 84.39% najlepszej solo)
- Hard voting: porównywalny lub nieco niższy od soft
- Nawet sam baseline obniża ensemble — warto pokazać wariant bez niego

## Pliki wynikowe

| Plik | Opis |
|---|---|
| `results/ensemble_summary.json` | Accuracy każdej metody + ensemble (soft/hard) + subset warianty |
| `plots/ensemble_comparison.png` | Wykres słupkowy porównawczy |
| `scripts/run_ensemble.py` | Skrypt ensemble |

---

# Phase 8 — Knowledge Distillation

**Status:** ⬜ Do zrobienia

**Wymaganie wstępne:** To samo co w Phase 7 — potrzebny checkpoint HPO modelu (`checkpoints/hpo_best_baseline_cnn.pt`) jako teacher oraz DARTS-derived genome (`results/darts_derived_genome.json`) do zbudowania studenta.

---

## Cel

Skompresować wiedzę z dużego modelu HPO (1.56M params, **teacher**, ~84% accuracy) do małego modelu DARTS-style (260K params, **student**) używając distillation loss. Pokazać, że student po dystylacji jest bliżej nauczyciela niż oryginalny DARTS.

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
  - Trenuje studenta z distillation loss przez 25-50 epok
  - Eksperymentuje z hiperparametrami distillation:
    - **Temperatura**: T = 1, 2, 4, 8
    - **Alpha**: α = 0.3, 0.5, 0.7
    - Zapisuje wyniki dla każdej kombinacji

- [ ] **Ewaluacja i porównanie**:
  - Accuracy studenta (distilled) vs studenta (oryginalny DARTS)
  - Accuracy studenta vs nauczyciela (HPO)
  - Ile udało się odzyskać z luki accuracy: `(acc_distilled - acc_original_darts) / (acc_teacher - acc_original_darts) * 100%`
  - Kompresja: 1.56M → 260K parametrów (~83% redukcji)

- [ ] **Wykresy**:
  - `plots/distillation_training_curves.png` — krzywe treningu studenta (loss, accuracy)
  - `plots/distillation_comparison.png` — słupki: teacher vs student_original vs student_distilled (najlepszy T/α)
  - `plots/distillation_temperature_sweep.png` — accuracy od temperatury (dla ustalonego α)

## Spodziewany wynik

| Model | Accuracy | Parametry |
|---|---|---|
| Teacher (HPO) | ~84.4% | 1.56M |
| Student (oryginalny DARTS) | ~78.9% | 260K |
| **Student distilled** | **~82-83%** | **260K** |
| Recovery rate | ~50-70% luki | — |

- Gain: ~3-4 pp nad oryginalnym DARTS
- Tylko ~1-2 pp straty do nauczyciela przy 83% mniejszym modelu
- Wyższa temperatura (T=4-8) powinna działać lepiej — bardziej miękki rozkład niesie więcej信息 o relacjach między klasami

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
