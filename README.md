# Consistency-Optimizers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

This repository contains the official implementation of the **Consistency Optimizer Family**, as introduced in the paper: *[The Consistency Optimizer Family: Modulating Parameter Updates via Local Trajectory Alignment]*.

This framework introduces a novel optimization approach that dynamically modulates parameter updates based on local trajectory alignment, explicitly coupling the update magnitude to the structural consistency of the optimization path. It includes two new optimizers: **ConsistencySGD** and **ConsistencyAdamW**.

## 📖 Table of Contents
* [Overview](#overview)
* [The Consistency Metric](#the-consistency-metric)
* [Algorithms](#algorithms)
* [Repository Structure](#repository-structure)
* [Experiments & Results](#experiments--results)
* [Citation](#citation)

## 🧠 Overview

First-order gradient-based optimization typically relies on exponential moving averages to smooth gradients and historical variance to scale learning rates. However, these methods structurally underutilize the element-wise directional agreement between the immediate gradient and accumulated momentum.

The **Consistency Optimizer Family** introduces a real-time, parameter-wise "consistency" metric. This rigorously bounded metric isolates pure geometric alignment, acting as a proxy for trajectory confidence:
* **High Consistency:** The gradient aligns with momentum. The optimizer systematically boosts the update step to accelerate convergence.
* **Low Consistency:** The gradient contradicts the momentum. The optimizer dampens the update, acting as a dynamic brake to prevent overshooting in noisy or oscillatory regions.

## 📐 The Consistency Metric

At the core of the framework is the Local Trajectory Consistency metric ($c_t$), which evaluates the normalized directional agreement between the current gradient ($g_t$) and the bias-corrected momentum ($\hat{m}_t$):

$$c_t = \frac{g_t}{|g_t| + \epsilon} \odot \frac{\hat{m}_t}{|\hat{m}_t| + \epsilon}$$

Because both vectors are normalized prior to element-wise multiplication, the metric is strictly bounded such that $c_t \in [-1, 1]^d$.

## ⚙️ Algorithms

### ConsistencySGD (Linearly Modulated Momentum)
ConsistencySGD applies a linear transformation to the consistency metric to scale the traditional momentum-driven step. It uses a tunable boost coefficient ($k$):

$$b_t = \mathbf{1} + k \cdot c_t$$

### ConsistencyAdamW (Exponentially Modulated Variance)
To cut through Adam's intrinsic variance-based normalization, ConsistencyAdamW applies an exponential scalar to the normalized step direction, ensuring a strictly positive, symmetric multiplicative modulation:

$$b_t = \exp(k \cdot c_t)$$

## 📂 Repository Structure

The repository contains Jupyter notebooks for reproducing the benchmark tests alongside the saved `.pkl` files containing historical run data and validation plots.

```text
Consistency-Optimizers/
├── California_Housing_saved_pkl_files_for_each_runs/  # Serialized run data for Tabular Regression
├── Final validation plots/                            # Exported loss/accuracy trajectory graphs
├── cifar10_saved_pkl_files_each_run/                  # Serialized run data for Computer Vision
├── imdb_saved_pkl_files_from_each_run/                # Serialized run data for NLP
├── California_Housing_Tests.ipynb                     # MLP training on California Housing
├── Cifar_10_Tests.ipynb                               # ResNet-18 training on CIFAR-10
├── Imdb_Tests.ipynb                                   # BiLSTM training on IMDB Reviews
└── README.md                                          # This file
```

Gemini said
Here is a comprehensive README.md file generated from your research paper and the repository structure shown in the image. You can copy and paste this directly into your GitHub repository.

Markdown
# Consistency-Optimizers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

This repository contains the official implementation of the **Consistency Optimizer Family**, as introduced in the paper: *[The Consistency Optimizer Family: Modulating Parameter Updates via Local Trajectory Alignment]*.

This framework introduces a novel optimization approach that dynamically modulates parameter updates based on local trajectory alignment, explicitly coupling the update magnitude to the structural consistency of the optimization path. It includes two new optimizers: **ConsistencySGD** and **ConsistencyAdamW**.

## 📖 Table of Contents
* [Overview](#overview)
* [The Consistency Metric](#the-consistency-metric)
* [Algorithms](#algorithms)
* [Repository Structure](#repository-structure)
* [Experiments & Results](#experiments--results)
* [Citation](#citation)

## 🧠 Overview

First-order gradient-based optimization typically relies on exponential moving averages to smooth gradients and historical variance to scale learning rates. However, these methods structurally underutilize the element-wise directional agreement between the immediate gradient and accumulated momentum.

The **Consistency Optimizer Family** introduces a real-time, parameter-wise "consistency" metric. This rigorously bounded metric isolates pure geometric alignment, acting as a proxy for trajectory confidence:
* **High Consistency:** The gradient aligns with momentum. The optimizer systematically boosts the update step to accelerate convergence.
* **Low Consistency:** The gradient contradicts the momentum. The optimizer dampens the update, acting as a dynamic brake to prevent overshooting in noisy or oscillatory regions.

## 📐 The Consistency Metric

At the core of the framework is the Local Trajectory Consistency metric ($c_t$), which evaluates the normalized directional agreement between the current gradient ($g_t$) and the bias-corrected momentum ($\hat{m}_t$):

$$c_t = \frac{g_t}{|g_t| + \epsilon} \odot \frac{\hat{m}_t}{|\hat{m}_t| + \epsilon}$$

Because both vectors are normalized prior to element-wise multiplication, the metric is strictly bounded such that $c_t \in [-1, 1]^d$.

## ⚙️ Algorithms

### ConsistencySGD (Linearly Modulated Momentum)
ConsistencySGD applies a linear transformation to the consistency metric to scale the traditional momentum-driven step. It uses a tunable boost coefficient ($k$):

$$b_t = \mathbf{1} + k \cdot c_t$$

### ConsistencyAdamW (Exponentially Modulated Variance)
To cut through Adam's intrinsic variance-based normalization, ConsistencyAdamW applies an exponential scalar to the normalized step direction, ensuring a strictly positive, symmetric multiplicative modulation:

$$b_t = \exp(k \cdot c_t)$$

## 📂 Repository Structure

The repository contains Jupyter notebooks for reproducing the benchmark tests alongside the saved `.pkl` files containing historical run data and validation plots.

```text
Consistency-Optimizers/
├── California_Housing_saved_pkl_files_for_each_runs/  # Serialized run data for Tabular Regression
├── Final validation plots/                            # Exported loss/accuracy trajectory graphs
├── cifar10_saved_pkl_files_each_run/                  # Serialized run data for Computer Vision
├── imdb_saved_pkl_files_from_each_run/                # Serialized run data for NLP
├── California_Housing_Tests.ipynb                     # MLP training on California Housing
├── Cifar_10_Tests.ipynb                               # ResNet-18 training on CIFAR-10
├── Imdb_Tests.ipynb                                   # BiLSTM training on IMDB Reviews
└── README.md                                          # This file
```
## 📊 Experiments & Results
The optimizers were benchmarked across three distinct deep learning domains against highly-tuned standard SGD-M and AdamW baselines. All experiments utilized a cosine annealing schedule with linear warmup.

1.  Tabular Regression (California Housing | MLP):
   ConsistencySGD achieved the absolute lowest MSE (0.2461±0.0024), outperforming standard SGD-M.ConsistencyAdamW (0.2484±0.0049) outperformed baseline AdamW.

2. Natural Language Processing (IMDB Sentiment | BiLSTM):
   ConsistencyAdamW yielded the highest overall performance (89.74%±0.46% accuracy), successfully translating trajectory alignment to sparse inputs and recurrent architectures.

4.  Computer Vision (CIFAR-10 | ResNet-18):
   ConsistencyAdamW (94.55%±0.02%) outperformed standard AdamW by +1.28%. ConsistencySGD achieved statistical parity with highly-tuned SGD-M configurations at an accelerated optimal learning rate.

## 📝 Citation
If you find this work useful in your research, please consider citing our paper:
```
@inproceedings{chhetri2026consistency,
  title={The Consistency Optimizer Family: Modulating Parameter Updates via Local Trajectory Alignment},
  author={Chhetri, Rajeeb Thapa and Thapa, Saurab and Kumar, Avinash and Chen, Zhixiong},
  booktitle={IEEE Conference (TBD)},
  year={2026},
  organization={Mercy University, School of Liberal Arts}
}
```
