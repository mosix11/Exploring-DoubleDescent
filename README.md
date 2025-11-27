<!-- # Exploring Double Descent

Code and experiments for exploring **double descent** in deep learning, inspired by:

- *Reconciling modern machine learning practice and the bias–variance trade-off* (Belkin et al.)
- *Deep Double Descent: Where Bigger Models and More Data Hurt* (Nakkiran et al.)

The goal of this repository is not just to reproduce individual figures, but to explore the phenomenon under different design choices (optimizer, loss, label noise, weight reuse, depth vs. width, synthetic vs. real data) and document what actually happens in practice.

---

## Overview

Double descent refers to the empirical observation that test error can first decrease (classical bias-variance regime), then increase around the interpolation threshold, and finally decrease again as model capacity grows.

In this repo I:

- Revisit the original experimental setups from Belkin et al. (fully connected networks on subsampled MNIST and CIFAR10) and Nakkiran et al. (CNNs and ResNets on CIFAR10/100 with label noise).
- Systematically vary the **optimizer**, **loss**, and **initialization/weight reuse scheme** to see when double descent appears or disappears.
- Build a **synthetic Gaussian mixture dataset** where label noise and class structure are fully controlled, and study:
  - how label noise shapes the interpolation threshold,
  - how weight reuse and freezing affect effective model capacity (EMC),
  - how **depth vs. width** at fixed parameter budget influences double descent.

All of these experiments are summarized in `report.pdf` along with observations and plots.

---

## Repository Structure

High-level layout:

- `src/`
  - Implementations of datasets, models, and trainers.
- `configs/`
  - This is where the experiments are defined. Each experiment is defined in a singla `yaml` file. I have put the ready to use file for some of the experiments. For others you can infer them from the current files.
- `train_model_wise.py`  
Varies **model capacity** (e.g. width) at fixed training setup to produce model-wise double descent curves (train/test loss & accuracy vs. number of parameters).
- `train_epoch_wise.py`  
Trains a fixed-capacity model for many epochs and logs curves over time to look for **epoch-wise double descent**.

---

## Implemented Experiments

### 1. Belkin fully-connected networks

**Datasets**

- Sub-sampled **MNIST**.
- Sub-sampled **CIFAR10** (including two class subsets, as in the original paper).

**Architectures**

- Fully connected net with one hidden layer.
- Model capacity is controlled via the **hidden width**, giving a smooth sweep across parameter counts around the interpolation threshold.

**What is explored**

- Original Belkin setup: SGD + MSE loss, weight reuse in the underparametrized regime.
- Removing weight reuse and using purely random initialization.
- Swapping **MSE ↔ CrossEntropy**.
- Swapping **SGD ↔ Adam** and changing training schedule.
- Adding controlled **label noise** (e.g. 20%) to study how the interpolation threshold moves.

The code and report show that some combinations (e.g. MSE + certain training setups) almost completely suppress double descent, while others (Adam + CrossEntropy) yield clear double descent around the expected parameter counts.

---

### 2. Nakkiran CNN & ResNet experiments

**Datasets**

- CIFAR10 with and without label noise (e.g. 20% symmetric noise).
- With and without data augmentation depending on the specific experiment.

**Architectures**

- A five layer CNN (CNN5) as described in *Deep Double Descent*.
- ResNet18 with varying base number of channels.

**What is explored**

- **Model-wise double descent** with CNN5 on CIFAR10:
  - Reimplementation of the “model-wise CNN5 CIFAR10” experiment.
- **Epoch-wise double descent**:
  - Training ResNet18 for thousands of epochs with Adam, comparing to the curves reported in Nakkiran et al.
  - Similar long training experiments with CNN5.
- Behaviour of **test loss vs. test accuracy** in the overfitting regime:
  - In some runs, test loss increases while test accuracy stays flat or even improves, leading to a mismatch between the two metrics and raising questions about what exactly “generalization degradation” means in practice.

These experiments highlight where the original paper’s behaviour is reproducible and where the dynamics differ, even with seemingly similar setups.

---

### 3. Synthetic Gaussian mixture dataset

To gain more control over the data distribution and label noise, the repo includes a synthetic dataset:

- High-dimensional input vectors drawn from a **mixture of Gaussians** (multimodal per class).
- A moderate number of classes.
- Precisely controlled **label noise** levels.

Using this controlled setting, several questions are investigated:

#### 3.1. Effect of label noise on double descent

- Vary the noise rate (including 0%, 20%, 50%, 75%).
- Track how the interpolation threshold moves and how the height and sharpness of the test-error peak change.
- Decompose the loss into contributions from **clean vs. noisy** samples to see which part of the data dominates the loss before and after interpolation.

#### 3.2. Weight reuse vs. random initialization

- **Random initialization**: each capacity is trained from scratch.
- **Weight reuse**: each wider model is initialized from the best checkpoint of the previous capacity (with new parameters added randomly).
- Comparison shows how weight reuse:
  - Alters the height/position of the double descent peak.
  - Changes whether the “best models” (under early stopping) are interpolating (zero training error) or not.
- A more extreme variant where reused weights are **frozen** and only newly added parameters are trainable is also explored, to see how a fixed function plus additional free parameters behaves around interpolation.

#### 3.3. Depth vs. width at fixed parameter budget

- Starting from a fully connected net with one hidden layer baseline, the code searches for deeper architectures with **approximately the same number of parameters**.
- For each depth:
  - Sweep total parameter count and measure train/test error, locating the interpolation threshold.
- This allows a direct comparison of how **depth changes effective model capacity** and how robust the resulting networks are to label noise, relative to simply making them wider.

---

## How to Use This Repository

> The exact commands and setup are intentionally left as placeholders so they can be adapted later.

### Installation & Environment

- **TODO:** Add environment creation steps (e.g. Conda/virtualenv) and how to install from `requirements.txt`.

### Datasets

- **TODO:** Document how to download / prepare MNIST and CIFAR-10.
- **TODO:** Explain how to configure and generate the synthetic Gaussian mixture dataset.

### Running Experiments

Each family of experiments has a corresponding entry script and set of configs:

- **Belkin-style FC experiments (MNIST / CIFAR-10)**
  - **TODO:** Document how to call `train_model_wise.py` and `train_model_wise_with_wreuse.py` with the right configs, and how to regenerate the plots.

- **Nakkiran-style CNN / ResNet experiments**
  - **TODO:** Document how to run `train_model_wise.py` / `train_epoch_wise.py` for CNN5 and ResNet-18, including noise and augmentation options.

- **Synthetic dataset & depth/width experiments**
  - **TODO:** Document how to generate the synthetic data and run `train_model_wise.py`, `compare_clean_noise_loss.py`, and `width_depth_comp.py` / `width_depth_plot.py`.

- **Plotting**
  - **TODO:** Show how to use `plot_model_wise.py`, `plot_model_wise_compare.py`, and `plot_comp_wr.py` to reproduce the figures from the report.

---

## Report

A detailed PDF report will be included in the repository (e.g. `report-double-descent-experiments-and-observations.pdf`).  
It contains:

- Full descriptions of the experimental setups.
- All figures (model-wise and epoch-wise curves, clean vs. noisy loss decompositions, width-vs-depth comparisons, etc.).
- Observations and hypotheses about when double descent does and does not appear, and what role label noise, optimization, and architecture play.

--- -->


# Exploring Double Descent

This repository contains a series of experiments investigating the **Double Descent** phenomenon in deep neural networks. The project aims to reproduce findings from foundational papers in the field while extending the scope to analyze the effects of initialization schemes (weight reuse and weight freezing), optimizer choices, loss functions, and network depth on the Effective Model Capacity (EMC).

A detailed PDF report containing all plots, experimental setups, and deeper analysis is included in this repository (`report.pdf`).

## 📖 Introduction

The "Double Descent" risk curve suggests that as model capacity increases, performance first improves, then degrades (underparameterized regime), and finally improves again (overparameterized regime). This project implements and scrutinizes the experiments presented in:

1.  **Reconciling modern machine learning practice and the bias-variance trade-off** (Belkin et al.)
2.  **Deep Double Descent: Where Bigger Models and More Data Hurt** (Nakkiran et al.)

Beyond reproduction, this repository explores **why** certain double descent curves appear (or fail to appear) depending on hyperparameter choices like the loss function (MSE vs. CrossEntropy) and initialization strategies.

## 🧪 Experiments & Key Observations

### 1. Revisiting Belkin et al. (MNIST-CIFAR10 & Fully Connected Networks)
We reproduce the standard double descent experiments using the MNIST and CIFAR10 datasets (subsampled as in the paper) and fully connected networks.
* **The Optimizer Factor:** We observed that the choice of optimizer significantly alters the visibility of the double descent peak. While vanilla SGD showed specific curves, switching to **Adam** made finding the interpolation threshold easier.
* **Loss Function Sensitivity:** A critical observation was made regarding **MSELoss** vs. **CrossEntropyLoss**. The experiments suggest that using MSELoss for classification tasks (as done in Belkin et al. experiments) can suppress the double descent curve compared to CrossEntropy, or alter the model's ability to find good solutions in the overparameterized regime.
* **Weight Reuse vs. Random Initialization:** We challenge the "Weight Reuse" initialization scheme (initializing larger models with parameters from smaller trained models). Our results indicate that while weight reuse creates a "fake" double descent curve, wherer the interpolation threshold can be moved arbitrarily.

### 2. Nakkiran et al. Reproduction (CIFAR-10 & CNNs)
We implemented experiments using **ResNet18** and **5-layer CNNs** on CIFAR10 with varying degrees of label noise.
* **Label Noise:** We confirm that adding label noise (e.g., 20%) shifts the interpolation threshold to a higher model capacity.
* **High Noise Robustness:** Interestingly, even with high levels of label corruption (50-75%), models were observed to cross very good solution thresholds in initial epochs, suggesting high robustness if early stopping is applied (as stated in the paper **SGD on Neural Networks Learns Functions of Increasing Complexity** (Nakkiran et al.)).

### 3. Extended Analysis: Weight Reuse & Freezing
To control noise levels precisely, we generated a **Synthetic Dataset** (Mixture of multimodal gaussians).
We conducted novel experiments to analyze the specific mechanics of **Weight Reuse**:
* **Consistent Weight Reuse:** We trained models sequentially, loading the best weights from the previous capacity.
* **Frozen Weights:** We attempted to "freeze" the reused weights, allowing the optimizer to only update the newly added parameters.
* **Observation:** When weights are frozen, there is a massive jump in test loss when the model moves from the first descent to the interpolation threshold. This suggests the optimizer treats the frozen function as a constant and aggressively overfits noise once capacity allows.

### 4. Depth vs. Width (Synthetic Dataset)
* **Objective:** Compare the Effective Model Capacity (EMC) of "Wide" networks vs. "Deep" networks with the same parameter count.
* **Observation:** Counterintuitively, deeper networks did not shift the interpolation threshold to the left as expected. Furthermore, deeper networks often exhibited higher test loss at the interpolation threshold compared to wider networks, suggesting a higher tendency to fit noise.


## Repository Structure

High-level layout:

- `src/`
  - Implementations of datasets, models, and trainers.
- `configs/`
  - This is where the experiments are defined. Each experiment is defined in a singla `yaml` file. I have put the ready to use file for some of the experiments. For others you can infer them from the current files.
- `train_model_wise.py`  
Varies **model capacity** (e.g. width) at fixed training setup to produce model-wise double descent curves (train/test loss & accuracy vs. number of parameters).
- `train_epoch_wise.py`  
Trains a fixed-capacity model for many epochs and logs curves over time to look for **epoch-wise double descent**.

<!-- --- -->


## How to Use This Repository

### Installation & Environment

Clone the repository and install the requirements by:
```bash
git clone https://github.com/mosix11/Exploring-DoubleDescent.git
cd Exploring-DoubleDescent
pip install -r requirements.txt
```

Then define the experiment you want to run in a `yaml` file inside `configs` directory following other experiments structure. 
To run an experiment sequentially, run:
```bash
python train_model_wise.py -c <config file name>
```
and if you want to train each model on multiple GPUs:
```bash
CUDA_VISIBLE_DEVICES=<gpu IDs> torchrun --nproc_per_node=<num processes> train_model_wise.py -c <config file name>
```

To run models in an experiment in parallel (not possible for experiments using `weight_reuse`), run:
```bash
python train_model_wise.py -c <config file name> --cpe <number of cpu cores for each experiment> --gpe <number of fraction of gpus for each experiment>
```

## References

```
@article{
doi:10.1073/pnas.1903070116,
author = {Mikhail Belkin  and Daniel Hsu  and Siyuan Ma  and Soumik Mandal },
title = {Reconciling modern machine-learning practice and the classical bias–variance trade-off},
journal = {Proceedings of the National Academy of Sciences},
volume = {116},
number = {32},
pages = {15849-15854},
year = {2019},
doi = {10.1073/pnas.1903070116},
URL = {https://www.pnas.org/doi/abs/10.1073/pnas.1903070116}}


@article{Nakkiran2019DeepDD,
  title={Deep double descent: where bigger models and more data hurt},
  author={Preetum Nakkiran and Gal Kaplun and Yamini Bansal and Tristan Yang and Boaz Barak and Ilya Sutskever},
  journal={Journal of Statistical Mechanics: Theory and Experiment},
  year={2019},
  volume={2021},
  url={https://api.semanticscholar.org/CorpusID:207808916}
}

@inbook{10.5555/3454287.3454601,
author = {Nakkiran, Preetum and Kaplun, Gal and Kalimeris, Dimitris and Yang, Tristan and Edelman, Benjamin L. and Zhang, Fred and Barak, Boaz},
title = {SGD on neural networks learns functions of increasing complexity},
year = {2019},
publisher = {Curran Associates Inc.},
address = {Red Hook, NY, USA},
booktitle = {Proceedings of the 33rd International Conference on Neural Information Processing Systems},
articleno = {314},
numpages = {11}
}

```

<!-- - Mikhail Belkin, Daniel Hsu, Siyuan Ma, Soumik Mandal.  
  **Reconciling modern machine learning practice and the bias–variance trade-off.**

- Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, Ilya Sutskever.  
  **Deep Double Descent: Where Bigger Models and More Data Hurt.** -->

<!-- ---

## License

- **TODO:** Add license information for this repository. -->
