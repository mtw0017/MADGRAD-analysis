CIFAR-10 & MNIST Optimizer Comparison
======================================

This project explores and benchmarks three optimization algorithms — MADGRAD, Adam, and SGD — on image classification tasks using convolutional neural networks.

Datasets Used
-------------
Dataset   | Samples | Image Size | Channels      | Classes
----------|---------|------------|---------------|---------
CIFAR-10  | 60,000  | 32x32      | RGB (3)       | 10
MNIST     | 70,000  | 28x28      | Grayscale (1) | 10

Purpose
-------
The goal is to compare how MADGRAD, a relatively new optimizer, performs against popular baselines (Adam, SGD) on dense vision datasets. Metrics tracked include training loss across epochs, with future extensions to validation accuracy.

Model Architecture
------------------
All models use a simple CNN with:
- Two convolutional layers + ReLU
- Max pooling
- Dropout for regularization
- Fully connected layers
- LogSoftmax output layer

Results
-------
CIFAR-10 (30 Epochs)
- Adam continues to lead in training loss minimization
- MADGRAD consistently outperforms SGD
- All optimizers show a steady decline — suggesting solid learning dynamics

MNIST (300 Epochs)
- On MNIST, all three optimizers converge, with MADGRAD and Adam reaching sub-1 loss quickly
- SGD lags initially but stabilizes well over time
- MADGRAD exhibits more noise but generalizes well with longer training

Installation & Setup
--------------------
pip install torch torchvision matplotlib
# If you're using a local clone of MADGRAD:
pip install -e ./madgrad

How to Run
----------
python testmad_dense.py     # for CIFAR-10
python testcmp.py           # for MNIST

Future Improvements
-------------------
- Track test/validation accuracy
- Add early stopping or learning rate scheduling
- Evaluate generalization to other datasets (e.g., FashionMNIST, SVHN)
- Export to notebook format for educational purposes
