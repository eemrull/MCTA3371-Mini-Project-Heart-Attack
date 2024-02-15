# MCTA 3371 Mini Project GUI: Genetic Algorithm and Neural Network

## Overview

This project is a mini application that combines Genetic Algorithms and Neural Network to predict Heart Risk problems.

| Matric Number | Name                                 |
| ------------- | ------------------------------------ |
| 2215359       | ARIF EMRULLAH BIN TAJUL ARIFFIN      |
| 2219537       | HAMZAH FAISAL                        |
| 2116281       | MUHAMMAD FADHLUL WAFI BIN AHMAD NAIM |
| 2110333       | MUHAMMAD NAUFAL BIN MOHAMMAD BAKRI   |

## Components

### GUI (PyQt5)

The GUI is implemented using PyQt5, a set of Python bindings for Qt libraries. It includes input boxes for users to specify parameters and a Matplotlib canvas to visualize the results.

### Genetic Algorithm

The Genetic Algorithm is responsible for evolving a population of potential solutions to the knapsack problem over several generations. It includes parameters such as population size, mutation rate, and crossover rate.

### Neural Network

The Neural Network is implemented using TensorFlow and Keras. It is trained on the provided heart risk dataset to predict the likelihood of an individual experiencing heart problems. The network architecture includes multiple layers with configurable activation functions.

### Files

    train.py: Script containing the training logic for the Genetic Algorithm and Neural Network.
    test.py: Script for testing the trained Neural Network.
    functions.py: Utility functions for data loading and processing.
    activations.py: Activation functions used in the Neural Network.
    nn.py: Neural Network classes and layers.
    data.csv: Sample dataset containing heart risk data.
    best_gene.pickle: Pickle file containing the trained Neural Network model.
