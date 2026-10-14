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
    3. Find bounding box of drawn pixels
    4. Crop to bounding box with padding
    5. Resize to 20x20 keeping aspect ratio
    6. Center in 28x28 canvas (MNIST style)
    7. Normalize to [0, 1]
    8. Flatten to 784-element list

    This mimics how MNIST digits are centered and sized.

    Args:
        img: PIL Image from the canvas.

    Returns:
        Tuple of (flattened_pixel_list, processed_image_for_display).
    """
    img = img.convert("L")  # grayscale
    img = ImageOps.invert(img)  # white digits on black bg
    arr = np.array(img)

    # Find bounding box of non-background pixels (threshold to avoid noise)
    coords = np.argwhere(arr > 30)
    if len(coords) == 0:
        # Empty canvas - return blank 28x28
        blank = np.zeros((28, 28), dtype=np.float32)
        return blank.flatten().tolist(), Image.fromarray((blank * 255).astype(np.uint8))

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add padding around the digit
    pad = 4
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(arr.shape[0], y_max + pad)
    x_max = min(arr.shape[1], x_max + pad)

    # Crop to bounding box
    cropped = arr[y_min:y_max, x_min:x_max]
    cropped_img = Image.fromarray(cropped)

    # Resize to fit in 20x20 keeping aspect ratio (MNIST digits are ~20x20 centered in 28x28)
    cropped_img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Center in 28x28 canvas
    final_img = Image.new("L", (28, 28), 0)
    px = (28 - cropped_img.width) // 2
    py = (28 - cropped_img.height) // 2
    final_img.paste(cropped_img, (px, py))

    # Normalize and flatten
    arr = np.array(final_img).astype(np.float32) / 255.0
    return arr.flatten().tolist(), final_img


# --- UI ---
st.title("Handwritten Digit Recognizer")
st.write("Draw a digit (0–9) in the canvas. The model expects MNIST-style input: centered, ~20×20 px, 2–3 px stroke width.")

# Session state for canvas reset
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Draw here")
    # Clear button
    if st.button("🗑️ Clear canvas", use_container_width=True):
        st.session_state.canvas_key += 1
        st.rerun()

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=25,
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
        return_image_data=True,
    )

with col_right:
    if canvas_result.image_data is not None:
        # Extract alpha channel as grayscale
        pil_img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))

        input_data, processed_img = preprocess(pil_img)

        # 28x28 preview with explanation
        st.subheader("What the model sees (28×28)")
        st.caption(
            "Your 280×280 drawing is cropped to the digit, resized to 20×20 "
            "keeping aspect ratio, centered in a 28×28 canvas, and normalized. "
            "This matches the MNIST training format."
        )
        st.image(
            processed_img.resize((140, 140), Image.NEAREST).convert("RGB"),
            caption="28×28 input (enlarged 5×)",
            width=140,
        )

        pred, probs = predict(input_data)

        # Prediction
        st.subheader(f"Prediction: **{pred}**")

        # Probability chart with explanation
        st.caption(
            "Bar chart shows the model's confidence for each digit (0–9). "
            "Higher bar = more confident. Values sum to 100%."
        )
        st.bar_chart({str(i): probs[i] for i in range(10)})
    else:
        st.info("Draw a digit on the left to see the prediction.")
