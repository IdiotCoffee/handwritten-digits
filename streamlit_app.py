import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import json
import math

# Load model weights
with open('model_weights.json', 'r') as f:
    model = json.load(f)

w_i_h = model['w_i_h']
w_h_o = model['w_h_o']
b_i_h = model['b_i_h']
b_h_o = model['b_h_o']

def softmax(predictions):
    m = max(predictions)
    exps = [math.exp(p - m) for p in predictions]
    total = sum(exps)
    return [e / total for e in exps]

def relu(x): return [max(0, xi) for xi in x]

def predict(input_flat):
    # Hidden layer
    pred_h = [sum(w * a for w, a in zip(weights, input_flat)) + b for weights, b in zip(w_i_h, b_i_h)]
    act_h = relu(pred_h)
    # Output layer
    pred_o = [sum(w * a for w, a in zip(weights, act_h)) + b for weights, b in zip(w_h_o, b_h_o)]
    act_o = softmax(pred_o)
    return act_o.index(max(act_o)), act_o

def preprocess(img: Image.Image):
    img = img.convert("L")                    # grayscale
    img = ImageOps.invert(img)               # white digits on black bg
    img = img.resize((28, 28))               # MNIST size
    arr = np.array(img).astype(np.float32)
    arr /= 255.0                             # normalize to [0, 1]
    return arr.flatten().tolist(), img       # also return image for debugging

# --- UI ---
st.title("🧠 Handwritten Digit Recognizer")
st.write("Draw a digit (0–9) below:")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="black",  # White strokes
    stroke_width=15,
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    pil_img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
    
    input_data, processed_img = preprocess(pil_img)
    st.image(processed_img.resize((140, 140)).convert("RGB"), caption="28x28 Input", width=140)
    

    pred, probs = predict(input_data)

    st.subheader(f"Prediction: **{pred}**")
    st.bar_chart(probs)
    
