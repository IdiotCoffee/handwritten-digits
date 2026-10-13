"""
Data loading utilities for the MNIST dataset.

This module provides functions to load training and test samples from CSV files
in the format expected by the neural network:
- Column 0: label (0-9)
- Columns 1-10: one-hot encoded target vector (10 values)
- Columns 11-794: 784 pixel values (28x28), normalized to [0, 1]
"""

import random
from pathlib import Path

# Data directory relative to this file
DATA_DIR = Path(__file__).parent / "data"


def get_training_samples(batch_size):
    """
    Generator that yields batches of training samples.

    Reads the entire training CSV, shuffles it, and yields batches of
    (labels, targets, inputs) where each is a list of length batch_size.

    Args:
        batch_size: Number of samples per batch.

    Yields:
        Tuple of (labels, targets, inputs) for each batch.
    """
    with open(DATA_DIR / "train.csv") as file:
        text = file.read()

    textlines = text.strip().split("\n")
    random.shuffle(textlines)
    start = 0
    while start < len(textlines):
        labels = []
        targets = []
        inputs = []
        end = start + batch_size
        for textline in textlines[start:end]:
            cells = textline.split(",")
            labels.append(int(cells[0]))
            # Columns 1-10: one-hot encoded target
            targets.append([float(c) for c in cells[1:11]])
            # Columns 11-794: 784 pixel values
            inputs.append([float(c) for c in cells[11:]])
        yield labels, targets, inputs
        start += batch_size


def get_test_samples():
    """
    Load all test samples at once (no batching needed for evaluation).

    Returns:
        Tuple of (labels, targets, inputs) for the entire test set.
    """
    with open(DATA_DIR / "test.csv", "r") as file:
        text = file.read()
    textlines = text.strip().split("\n")
    labels = []
    targets = []
    inputs = []
    for textline in textlines:
        cells = textline.split(",")
        labels.append(int(cells[0]))
        targets.append([float(c) for c in cells[1:11]])
        inputs.append([float(c) for c in cells[11:]])
    return labels, targets, inputs


def plot_number(inputs):
    """
    Print a 28x28 digit as ASCII art for debugging.

    Maps pixel values (0-1) to characters: .░▒▓█
    """
    line = ""
    for p in inputs:
        line += ".░▒▓█"[round(p*4)]
        if len(line) > 27:
            print(line)
            line=""
