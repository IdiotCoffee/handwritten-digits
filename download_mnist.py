"""
Download and prepare the full MNIST dataset.

This script downloads the complete MNIST dataset (60,000 training + 10,000 test images)
and converts it to the CSV format used by this project:
- Column 0: label (0-9)
- Columns 1-10: one-hot encoded target vector
- Columns 11-794: 784 pixel values (28x28), normalized to [0, 1]

The resulting files are saved to the data/ directory.
"""

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Primary mirrors (Google Cloud Storage is reliable)
MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

# Fallback mirrors if primary fails
MNIST_MIRRORS = {
    "train_images": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    ],
    "train_labels": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    ],
    "test_images": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    ],
    "test_labels": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
    ],
}


def download_file(url: str, dest: Path, mirrors: list[str] | None = None):
    """Download a file if it doesn't already exist. Tries mirrors on failure."""
    if dest.exists():
        print(f"Already exists: {dest.name}")
        return

    urls = [url] + (mirrors or [])
    last_error = None

    for i, u in enumerate(urls):
        try:
            if i == 0:
                print(f"Downloading {dest.name}...")
            else:
                print(f"  Trying mirror {i}...")
            urllib.request.urlretrieve(u, dest)
            print(f"Downloaded: {dest.name}")
            return
        except urllib.error.HTTPError as e:
            last_error = e
            if i < len(urls) - 1:
                print(f"  Failed (HTTP {e.code}), trying next mirror...")
            continue
        except urllib.error.URLError as e:
            last_error = e
            if i < len(urls) - 1:
                print(f"  Failed ({e.reason}), trying next mirror...")
            continue
        except OSError as e:
            last_error = e
            if i < len(urls) - 1:
                print(f"  Failed ({e}), trying next mirror...")
            continue

    raise RuntimeError(f"Failed to download {dest.name} from all sources: {last_error}")


def read_idx3_ubyte(filepath: Path) -> np.ndarray:
    """Read IDX3 format (images). Returns array of shape (n_samples, 784)."""
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Invalid magic number {magic} for images"
        assert rows == 28 and cols == 28, f"Expected 28x28, got {rows}x{cols}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols).astype(np.float32) / 255.0


def read_idx1_ubyte(filepath: Path) -> np.ndarray:
    """Read IDX1 format (labels). Returns array of shape (n_samples,)."""
    with gzip.open(filepath, "rb") as f:
        magic, _ = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Invalid magic number {magic} for labels"
        return np.frombuffer(f.read(), dtype=np.uint8)


def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer labels to one-hot encoded vectors."""
    n = len(labels)
    one_hot = np.zeros((n, num_classes), dtype=np.float32)
    one_hot[np.arange(n), labels] = 1.0
    return one_hot


def write_csv(images: np.ndarray, labels: np.ndarray, targets: np.ndarray, output_path: Path):
    """Write data to CSV format expected by digitreader.py."""
    print(f"Writing {len(images)} samples to {output_path.name}...")
    with open(output_path, "w") as f:
        for i in range(len(images)):
            label = labels[i]
            target_str = ",".join(f"{v:.1f}" for v in targets[i])
            pixel_str = ",".join(f"{v:.6f}" for v in images[i])
            f.write(f"{label},{target_str},{pixel_str}\n")
    print(f"Done: {output_path.name}")


def main():
    print("Downloading MNIST dataset...")

    # Download all files with mirror fallback
    for name, url in MNIST_URLS.items():
        dest = DATA_DIR / f"{name}.gz"
        mirrors = MNIST_MIRRORS.get(name, [])
        download_file(url, dest, mirrors)

    print("\nProcessing training data...")
    train_images = read_idx3_ubyte(DATA_DIR / "train_images.gz")
    train_labels = read_idx1_ubyte(DATA_DIR / "train_labels.gz")
    train_targets = one_hot_encode(train_labels)
    write_csv(train_images, train_labels, train_targets, DATA_DIR / "train.csv")

    print("\nProcessing test data...")
    test_images = read_idx3_ubyte(DATA_DIR / "test_images.gz")
    test_labels = read_idx1_ubyte(DATA_DIR / "test_labels.gz")
    test_targets = one_hot_encode(test_labels)
    write_csv(test_images, test_labels, test_targets, DATA_DIR / "test.csv")

    # Clean up downloaded gz files
    for name in MNIST_URLS:
        gz_file = DATA_DIR / f"{name}.gz"
        if gz_file.exists():
            gz_file.unlink()

    print("\nMNIST dataset ready!")
    print(f"Training samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    print(f"Files saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
