
# Exploring Double Descent

This repository contains a series of experiments investigating the **Double Descent** phenomenon in deep neural networks. The project aims to reproduce findings from foundational papers in the field while extending the scope to analyze the effects of initialization schemes (weight reuse and weight freezing), optimizer choices, loss functions, and network depth on the Effective Model Capacity (EMC).

A detailed PDF report containing all plots, experimental setups, and deeper analysis is included in this repository (`report.pdf`).

## Introduction

The "Double Descent" risk curve suggests that as model capacity increases, performance first improves, then degrades (underparameterized regime), and finally improves again (overparameterized regime). This project implements and scrutinizes the experiments presented in:

1.  **Reconciling modern machine learning practice and the bias-variance trade-off** (Belkin et al.)
2.  **Deep Double Descent: Where Bigger Models and More Data Hurt** (Nakkiran et al.)

Beyond reproduction, this repository explores **why** certain double descent curves appear (or fail to appear) depending on hyperparameter choices like the loss function (MSE vs. CrossEntropy) and initialization strategies.

## Experiments & Key Observations

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
