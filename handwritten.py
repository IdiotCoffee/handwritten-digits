"""
Training script for a from-scratch neural network on MNIST.

This implements a 2-layer feedforward neural network (784 -> 128 -> 10)
trained with mini-batch gradient descent and backpropagation.
No deep learning frameworks — just Python, NumPy (for arrays), and math.
"""

import json
import math
import random
from pathlib import Path

import digitreader as reader

# Weights directory relative to this file
WEIGHTS_DIR = Path(__file__).parent / "weights"


def softmax(predictions):
    """
    Compute softmax activation for a list of logits.

    Subtracts max for numerical stability before exponentiating.

    Args:
        predictions: List of raw output scores (logits).

    Returns:
        List of probabilities summing to 1.
    """
    m = max(predictions)
    temp = [math.exp(p - m) for p in predictions]
    total = sum(temp)
    return [t / total for t in temp]


def log_loss(activations, targets):
    """
    Compute binary cross-entropy loss for one sample.

    Args:
        activations: Predicted probabilities (after softmax).
        targets: One-hot encoded ground truth.

    Returns:
        Scalar loss value.
    """
    losses = [-t * math.log(a) - (1 - t) * math.log(1 - a) for a, t in zip(activations, targets)]
    return sum(losses)


# Hyperparameters
epochs = 21
batch_size = 128
learning_rate = 0.1
# Network architecture: 784 input (28x28), 128 hidden, 10 output (digits 0-9)
input_count, hidden_count, output_count = 784, 128, 10

# Initialize weights with small random values centered at 0
# w_i_h: input-to-hidden weights [hidden_count x input_count]
# w_h_o: hidden-to-output weights [output_count x hidden_count]
w_i_h = [[random.random() - 0.5 for _ in range(input_count)] for _ in range(hidden_count)]
w_h_o = [[random.random() - 0.5 for _ in range(hidden_count)] for _ in range(output_count)]
# Biases initialized to 0
b_i_h = [0 for _ in range(hidden_count)]
b_h_o = [0 for _ in range(output_count)]

"""Training loop with mini-batch gradient descent."""

for epoch in range(epochs):
    for labels, targets, inputs in reader.get_training_samples(batch_size):
        # --- Forward pass ---
        # Hidden layer pre-activations: w_i_h @ input + b_i_h
        pred_h = [
            [
                sum([w * a for w, a in zip(weights, inp)]) + bias
                for weights, bias in zip(w_i_h, b_i_h)
            ]
            for inp in inputs
        ]
        # ReLU activation
        act_h = [[max(0, p) for p in pred] for pred in pred_h]
        # Output layer pre-activations: w_h_o @ act_h + b_h_o
        pred_o = [
            [
                sum([w * a for w, a in zip(weights, inp)]) + bias
                for weights, bias in zip(w_h_o, b_h_o)
            ]
            for inp in act_h
        ]
        # Softmax activation
        act_o = [softmax(predictions) for predictions in pred_o]

        # --- Loss computation ---
        cost = sum([log_loss(a, t) for a, t in zip(act_o, targets)]) / len(act_o)
        print(f"epoch:{epoch} cost:{cost:.4f}")

        # --- Backpropagation ---
        # Output layer error: derivative of cross-entropy + softmax = prediction - target
        errors_d_o = [[a - t for a, t in zip(ac, ta)] for ac, ta in zip(act_o, targets)]

        # Transpose w_h_o for backprop to hidden layer
        w_h_o_T = list(zip(*w_h_o))
        # Hidden layer error: (w_h_o_T @ errors_d_o) * ReLU'(pred_h)
        # ReLU derivative is 1 if pred > 0 else 0
        errors_d_h = [
            [
                sum([d * w for d, w in zip(deltas, weights)]) * (0 if p <= 0 else 1)
                for weights, p in zip(w_h_o_T, pred)
            ]
            for deltas, pred in zip(errors_d_o, pred_h)
        ]

        # --- Gradient computation ---
        # Transpose activations for matrix multiplication
        act_h_T = list(zip(*act_h))
        errors_d_o_T = list(zip(*errors_d_o))
        errors_d_h_transpose = list(zip(*errors_d_h))
        inputs_transpose = list(zip(*inputs))

        # Weight gradients: delta @ activation.T
        w_h_o_d = [
            [sum([d * a for d, a in zip(deltas, act)]) for deltas in errors_d_o_T]
            for act in act_h_T
        ]
        b_h_o_d = [sum([d for d in deltas]) for deltas in errors_d_o_T]

        w_i_h_d = [
            [sum([d * a for d, a in zip(deltas, act)]) for deltas in errors_d_h_transpose]
            for act in inputs_transpose
        ]
        b_i_h_d = [sum([d for d in deltas]) for deltas in errors_d_h_transpose]

        # --- Weight updates ---
        # Average gradients over batch and apply learning rate
        w_h_o_d_transpose = list(zip(*w_h_o_d))
        for y in range(output_count):
            for x in range(hidden_count):
                w_h_o[y][x] -= learning_rate * w_h_o_d_transpose[y][x] / len(inputs)
            b_h_o[y] -= learning_rate * b_h_o_d[y] / len(inputs)

        w_i_h_d_transpose = list(zip(*w_i_h_d))
        for y in range(hidden_count):
            for x in range(input_count):
                w_i_h[y][x] -= learning_rate * w_i_h_d_transpose[y][x] / len(inputs)
            b_i_h[y] -= learning_rate * b_i_h_d[y] / len(inputs)

    # --- Evaluation on test set after each epoch ---
    labels, targets, inputs = reader.get_test_samples()
    pred_h = [
        [sum([w * a for w, a in zip(weights, inp)]) + bias for weights, bias in zip(w_i_h, b_i_h)]
        for inp in inputs
    ]
    act_h = [[max(0, p) for p in pre] for pre in pred_h]
    pred_o = [
        [sum([w * a for w, a in zip(weights, inp)]) + bias for weights, bias in zip(w_h_o, b_h_o)]
        for inp in act_h
    ]
    act_o = [softmax(predictions) for predictions in pred_o]
    correct = 0
    for a, t, i in zip(act_o, targets, inputs):
        ma_neuron = a.index(max(a))
        ma_target = t.index(max(t))
        if ma_neuron == ma_target:
            correct += 1

    print(f"correct: {correct}/{len(act_o)} ({correct / len(act_o):%})")


# Save trained weights to JSON
model_data = {"w_i_h": w_i_h, "w_h_o": w_h_o, "b_i_h": b_i_h, "b_h_o": b_h_o}
with open(WEIGHTS_DIR / "model_weights.json", "w") as f:
    json.dump(model_data, f)
