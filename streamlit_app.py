"""
Streamlit web interface for handwritten digit recognition.

Loads the trained model weights and provides a drawable canvas for users
to draw digits, which are then classified in real-time.
"""

import json
import math
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load model weights from the weights directory
WEIGHTS_DIR = Path(__file__).parent / "weights"
with open(WEIGHTS_DIR / "model_weights.json", "r") as f:
    model = json.load(f)

w_i_h = model["w_i_h"]  # Input-to-hidden weights [128 x 784]
w_h_o = model["w_h_o"]  # Hidden-to-output weights [10 x 128]
b_i_h = model["b_i_h"]  # Hidden layer biases [128]
b_h_o = model["b_h_o"]  # Output layer biases [10]


def softmax(predictions):
    """Numerically stable softmax."""
    m = max(predictions)
    exps = [math.exp(p - m) for p in predictions]
    total = sum(exps)
    return [e / total for e in exps]


def relu(x):
    """Rectified Linear Unit activation."""
    return [max(0, xi) for xi in x]


def predict(input_flat):
    """
    Run forward pass through the network.

    Args:
        input_flat: Flattened 784-element list of pixel values [0, 1].

    Returns:
        Tuple of (predicted_digit, probability_distribution).
    """
    # Hidden layer: linear + ReLU
    pred_h = [
        sum(w * a for w, a in zip(weights, input_flat)) + b for weights, b in zip(w_i_h, b_i_h)
    ]
    act_h = relu(pred_h)
    # Output layer: linear + softmax
    pred_o = [sum(w * a for w, a in zip(weights, act_h)) + b for weights, b in zip(w_h_o, b_h_o)]
    act_o = softmax(pred_o)
    return act_o.index(max(act_o)), act_o


def preprocess(img: Image.Image):
    """
    Preprocess a drawn image to match MNIST format.

    Steps:
    1. Convert to grayscale
    2. Invert (MNIST has white digits on black background)
    3. Resize to 28x28
    4. Normalize to [0, 1]
    5. Flatten to 784-element list

    Args:
        img: PIL Image from the canvas.

    Returns:
        Tuple of (flattened_pixel_list, processed_image_for_display).
    """
    img = img.convert("L")  # grayscale
    img = ImageOps.invert(img)  # white digits on black bg
    img = img.resize((28, 28))  # MNIST size
    arr = np.array(img).astype(np.float32)
    arr /= 255.0  # normalize to [0, 1]
    return arr.flatten().tolist(), img  # also return image for debugging


# --- UI ---
st.title("Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) below:")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="black",  # White strokes on white background
    stroke_width=15,
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
    return_image_data=True,
)

if canvas_result.image_data is not None:
    # Extract alpha channel as grayscale
    pil_img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))

    input_data, processed_img = preprocess(pil_img)
    # Display the 28x28 processed image (enlarged for visibility)
    st.image(processed_img.resize((140, 140)).convert("RGB"), caption="28x28 Input", width=140)

    pred, probs = predict(input_data)

    st.subheader(f"Prediction: **{pred}**")
    st.bar_chart(probs)
