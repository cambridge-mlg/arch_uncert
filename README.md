# Variational Depth Search in ResNets
​
[![arxiv](https://img.shields.io/badge/stat.ML-arXiv%3A2002.02797-B31B1B.svg)](https://arxiv.org/abs/2002.02797)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-376/)
[![Pytorch 1.3](https://img.shields.io/badge/pytorch-1.3.1-blue.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/cambridge-mlg/arch_uncert/blob/master/LICENSE)
​
<!--[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/LICENSE) -->
​
​
One-shot neural architecture search allows joint learning of weights and network architecture, reducing computational cost. We limit our search space to the depth of residual networks and formulate an analytically tractable variational objective that allows for obtaining an unbiased approximate posterior over depths in one-shot. We propose a heuristic to prune our networks based on this distribution. We compare our proposed method against manual search over network depths on the MNIST, Fashion-MNIST, SVHN datasets. We find that pruned networks do not incur a loss in predictive performance, obtaining accuracies competitive with unpruned networks. Marginalising over depth allows us to obtain better-calibrated test-time uncertainty estimates than regular networks, in a single forward pass.
​
### Requirements
Python packages:
* test-tube 0.7.5
* Pytorch 1.3.1, torchvision 0.4.2
* Numpy 1.17.4
* Matplotlib 3.1.2
* scikit-learn 0.22
* scypy 1.3.3
​
## Running Experiments from the Paper
​
Integers passed as an argument to python scripts represent which cuda device
to use. If you only have one GPU, use 0. If you dont have a GPU, pass any integer and your CPU
will be used automatically.
​
First change into experiments directory:
```bash
cd experiments
```
​
### Spirals
​
In order to reproduce the plots from our paper exactly, you will need to run
each script multiple times. Different runs of each experiment are automatically saved
separately. 
​
Train Learnt Depth Networks with every maximum depth:
```bash
python scan_max_depth_spirals.py 0
```
​
Train deterministic depth networks of every depth:
```bash
python scan_deterministic_depth_spirals.py 0
```
​
Train Learnt Depth Networks with different dataset sizes:
```bash
python scan_data_amount.py 0
```
​
Train Learnt Depth Networks with different dataset complexity:
```bash
python scan_spiral_complexity.py 0
```
​
Train Learnt Depth Networks with different widths:
```bash
python scan_width_spirals.py 0
```
​
### Images
​
Each script runs experiments on all three datasets (MNIST, Fashion-MNIST and SVHN)
and repeats each experiment 4 times.
​
Train Learnt Depth Networks:
```bash
python scan_max_depth_images.py 0
```
​
Train Deterministic Depth Networks:
```bash
python scan_deterministic_depth_images.py 0
```
​
## Generate Plots from Paper
​
All plotting code is contained within the notebooks in the ./notebooks/ folder.
Once the experiments have been run, running the notebooks will generate the plots.
​
## Citation
If you find this repo useful, please cite:
​
```bibtex
@misc{antoran2020variational,
    title={Variational Depth Search in ResNets},
    author={Javier Antorán and James Urquhart Allingham and José Miguel Hernández-Lobato},
    year={2020},
    eprint={2002.02797},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
```
