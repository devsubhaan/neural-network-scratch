# Neural Network From Scratch
 
A multi-layer perceptron implemented from scratch in Python using NumPy, without deep learning frameworks such as PyTorch or TensorFlow.
 
**Status:** work in progress. The mathematical foundations, activation functions and loss are implemented and tested. Layers, gradient checking and training are under way.
 
---
 
## Overview
 
The aim of this project is to understand how neural networks learn by building every piece by hand: the forward pass, the loss, backpropagation and the optimisers. Each equation is derived on paper first (see [`docs/derivations.md`](docs/derivations.md)), then implemented in NumPy and checked with unit tests.
 
The planned end state is a configurable network trained on XOR and MNIST, with its gradients verified numerically and its results compared against an equivalent PyTorch model.
 
---
 
## Features
 
Core components:
 
- [x] Activation functions with derivatives (ReLU, sigmoid, softmax)
- [x] Numerically stable softmax cross-entropy loss and its gradient
- [x] Backpropagation derived by hand (scalar, vector and batch forms) in [`docs/derivations.md`](docs/derivations.md)
- [x] Unit tests for the activations and the loss (pytest)
- [ ] Weight initialisation: He and Xavier (in progress)
- [ ] Dense layer: forward pass (in progress)
- [ ] Dense layer: backward pass
- [ ] ReLU layer
- [ ] Numerical gradient checking
- [ ] Sequential network class
Training:
 
- [ ] Mini-batch gradient descent training loop
- [ ] Optimisers: SGD with momentum and Adam
- [ ] Metrics: accuracy, confusion matrix, precision and recall
Experiments:
 
- [ ] XOR sanity check with decision boundary plot
- [ ] MNIST baseline
- [ ] Comparison of activations, initialisation schemes and optimisers
- [ ] Comparison against an equivalent PyTorch model
- [ ] Speed benchmark
---
 
## Mathematical Background
 
The full derivations are in [`docs/derivations.md`](docs/derivations.md):
 
- Backpropagation as the chain rule, for a scalar network with worked numbers
- The vector form, with the shape rule that fixes where each transpose goes
- Softmax and cross-entropy, including the gradient p - y
- Batch equations used by the implementation
- Numerical gradient checking with central differences
---
 
## Project Structure
 
```
neural-net-from-scratch/
├── src/nn_scratch/
│   ├── activations.py     # sigmoid, ReLU, softmax and their derivatives
│   ├── losses.py          # softmax cross-entropy and its gradient
│   ├── initializers.py    # He and Xavier initialisation (in progress)
│   └── layers.py          # Layer base class and Dense (in progress)
├── tests/                 # pytest unit tests
├── docs/
│   └── derivations.md     # hand-derived backpropagation
└── notebooks/
    └── matrix.ipynb       # worked matrix example
```
 
Planned: `network.py`, `optimizers.py`, `gradient_check.py` and an `experiments/` folder.
 
---
 
## Installation
 
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```
 
---
 
## Usage
 
Training is not implemented yet. This section will show how to train and evaluate the network once the training loop is complete.
 
---
 
## Experiments and Results
 
No results yet. The planned experiments are listed under Features. This section will record the dataset, baselines, metrics and results for each one.
 
---
 
## Tests
 
```bash
.venv/bin/python -m pytest
```
 
Current tests cover the activation functions, their derivatives and the loss.
 
---
 
## Future Improvements
 
- Convolutional layers
- Regularisation (L2 and dropout) and batch normalisation
- A small transformer built on the same layer interface
---
