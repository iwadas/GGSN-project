# Neural Architecture Search & Hyperparameter Optimization Research Project

## Overview

This repository contains a research-oriented project focused on the automation of deep neural network design using:

- Hyperparameter Optimization (HPO)
- Neural Architecture Search (NAS)
- Evolutionary Algorithms
- Differentiable Architecture Search (DARTS-inspired methods)
- Hardware-Aware Optimization

The project investigates how automated methods can design efficient convolutional neural networks (CNNs) for image classification tasks while balancing:

- Classification accuracy
- Model size
- Inference latency

The experiments are conducted primarily on the CIFAR-10 dataset using Google Colab GPU resources.

---

# Research Goal

The main objective of this project is to compare different automated deep learning optimization strategies and evaluate their effectiveness in discovering efficient CNN architectures.

The project focuses on multi-objective optimization:

```math
Fitness = Accuracy - \alpha \cdot Params - \beta \cdot Latency