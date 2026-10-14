# Handwritten Digit Recognizer

A neural network built from scratch that recognizes handwritten digits drawn in a web browser. No deep learning frameworks — just Python, NumPy, and the math behind backpropagation. The interface is built with Streamlit.

## What This Project Is

This started as a learning exercise while following a Udemy course on building neural networks from scratch, but evolved into a project that actually works - neural networks under the hood are trained to draw the number that a user draws on the canvas. The goal was to understand how log loss, softmax, ReLU, and backpropagation actually work under the hood, without hiding behind library abstractions.

The network architecture is a simple two-layer feedforward network: 784 input neurons (one per pixel of a 28×28 image), 128 hidden neurons with ReLU activation, and 10 output neurons with softmax for classification. Training uses mini-batch gradient descent with a decaying learning rate.

The web interface lets you draw a digit on a 280×280 canvas. The drawing is preprocessed to match the MNIST format — converted to grayscale, inverted (MNIST uses white-on-black), resized to 28×28, normalized, and flattened — then passed through the network for a real-time prediction with a probability bar chart.

## Project Structure

```
handwritten-digits/
├── data/                 # Training and test data (gitignored)
│   ├── train.csv         # 60,000 training samples (after running download_mnist.py)
│   └── test.csv          # 10,000 test samples
├── weights/              # Model weights (gitignored)
│   └── model_weights.json
├── download_mnist.py     # Script to fetch full MNIST and convert to CSV
├── digitreader.py        # Data loading utilities
├── handwritten.py        # Training script (run this to train)
├── streamlit_app.py      # Web interface (run with: streamlit run streamlit_app.py)
├── pyproject.toml        # UV project configuration
└── uv.lock               # Locked dependencies
```

## Getting Started

### Prerequisites

- Python 3.10+
- [UV](https://docs.astral.sh/uv/) for dependency management

### Installation

```bash
# Clone the repo
git clone https://github.com/IdiotCoffee/handwritten-digits
cd handwritten-digits

# Install dependencies with UV
uv sync
```

### Download the Full MNIST Dataset

The repository only contains a small subset of MNIST by default. To get the full 60,000 training / 10,000 test images:

```bash
uv run python download_mnist.py
```

This downloads the original MNIST binary files, converts them to the CSV format used by `digitreader.py`, and saves them to `data/`. The CSV files are gitignored since they're large and reproducible.

### Train the Model

```bash
uv run python handwritten.py
```

This runs 10 epochs of mini-batch gradient descent (batch size 64, initial learning rate 0.1 with step decay ×0.5 every 7 epochs) and prints the average loss, current learning rate, and test accuracy after each epoch. The trained weights are saved to `weights/model_weights.json`.

**Training time:** ~5–8 minutes on a modern CPU for 21 epochs over 60,000 samples (no GPU, pure Python loops). Each epoch processes ~469 batches (~128 samples each). The pure-Python implementation is slower than NumPy-vectorized or GPU training.

**Learning rate schedule:** 0.1 (epochs 0–6) → 0.05 (epochs 7–13) → 0.025 (epochs 14–20). This helps converge to a sharper minimum.

**Loss tracking placeholder:**

| Run | Initial Loss (epoch 0) | Final Loss (epoch 10) | Final Test Accuracy |
|-----|------------------------|----------------------|---------------------|
| 1   | 4.091                     | 0.235                    | 95.15%                   |

**How the loss is calculated:** The loss is binary cross-entropy (log loss) averaged over each batch. For each sample, with predicted probabilities `a` (after softmax) and one-hot target `t`:

```
loss = -Σ[t_i * log(a_i) + (1 - t_i) * log(1 - a_i)]  for i = 0..9
```

This penalizes confident wrong predictions heavily. The epoch loss reported is the mean of batch losses.

### Run the Web App

```bash
uv run streamlit run streamlit_app.py
```

Open the URL shown in the terminal (usually http://localhost:8501) and start drawing digits.

## Using Your Own Data

The network expects data in a specific CSV format:

- Column 0: integer label (0–9)
- Columns 1–10: one-hot encoded target vector (10 floats, e.g., `0,0,0,0,0,1,0,0,0,0` for digit 5)
- Columns 11–794: 784 pixel values (28×28), normalized to [0, 1]

To train on your own dataset:

1. Prepare a CSV file matching this format.
2. Place it in `data/` as `train.csv` (and optionally `test.csv`).
3. Run `handwritten.py`.

The `digitreader.py` module handles batching and shuffling automatically. If your data has different dimensions, you'll need to adjust `input_count` in `handwritten.py` and the preprocessing in `streamlit_app.py`.

## Known Limitations

- The network is shallow (one hidden layer) and uses simple SGD. Modern architectures (CNNs, deeper networks, better optimizers) achieve much higher accuracy.
- Training on the full MNIST takes a few minutes on a modern CPU. No GPU acceleration is used.
- The Streamlit canvas captures mouse/touch input at 280×280, which is then downscaled. Very thin or very thick strokes may not match the stroke weight MNIST models expect.
- Digits like 7 and 9 are sometimes confused depending on drawing style.

## Why This Exists

I built this to demystify what happens inside `model.fit()` — to see the matrix multiplications, the chain rule in action, and how a bunch of random numbers gradually become a digit classifier. If you've only ever used PyTorch or TensorFlow, writing the backward pass by hand is a useful exercise.
