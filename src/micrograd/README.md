# Micrograd (Python Implementation)

A tiny scalar-valued auto-gradient engine and a neural network built from scratch in Python, following Andrej Karpathy's tutorial.

## 🧠 Acknowledgments & Attribution

This project is an implementation on **Andrej Karpathy's** video:
* **Title:** *The spelled-out intro to neural networks and backpropagation: building micrograd*
* **Watch the video:** [YouTube Link](https://www.youtube.com/watch?v=VMj-3S1tku0)
* **Original Repository:** [karpathy/micrograd](https://github.com/karpathy/micrograd)

All core concepts, architecture design, and backpropagation mechanics are credit to him. This folder was an exercise to understand micrograd and backpropagation a lot more.
---

## 🚀 About the Project

`micrograd` implements a backpropagation engine over a DAG (Directed Acyclic Graph) of scalar values. It computes gradients with respect to all operations (`+`, `-`, `*`, `/`, `**`, `ReLU`, `tanh`), allowing us to build multi-layer perceptrons (MLPs) and train them.

## ✨ Features

* **Scalar Autograd Engine:** Tracks operations to build a computation graph and automatically compute gradients via the chain rule.
* **Neural Network API:** Lightweight `Value`, `Layer`, `Neuron`, and `MLP` classes mimicking a mini-PyTorch.