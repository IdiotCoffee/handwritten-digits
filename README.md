# Handwritten Digits Recognizer

This project is a handwritten digits recognizer built using neural networks. The implementation is based on a Udemy course, and it uses minimal external libraries to ensure a deeper understanding of the underlying mathematics.

## Overview

The model is trained on the MNIST database, a widely-used dataset for handwritten digit classification. The only external library used in this project is Pillow, which is utilized for drawing and visualizing the digits.

### Model Architecture

- **Input Layers**: 784 nodes (corresponding to the 28x28 pixel images)
- **Hidden Layers**: 20 nodes
- **Output Layers**: 10 nodes (one for each digit from 0 to 9)

### Training Details

- **Epochs**: 6
- **Learning Rate**: 1
- **Accuracy**: 89% after 6 epochs

## Mathematical Concepts

The project involves several mathematical functions and concepts, including:

- **Log Loss**: A loss function commonly used in classification tasks.
- **Softmax**: A function that converts raw model outputs into probability distributions.

## Dependencies

- **Pillow**: Used for drawing the digits.

Install it using pip:

```bash
pip install Pillow
```

[Udemy Course Link]( https://www.udemy.com/course/build-neural-networks-from-scratch-with-python-step-by-step)
