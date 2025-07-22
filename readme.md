# 🧠 Handwritten Digit Recognizer

A simple neural network, trained completely from scratch, that predicts the digit you draw on a canvas. No deep learning frameworks — just Python, NumPy, and math. The web interface is built using [Streamlit](https://streamlit.io).


The model attained a 91.8% accuracy during simple training using the database.
---

## ✨ What This Is

- A **neural network built entirely from scratch**, following [this Udemy course](https://www.udemy.com/course/build-neural-networks-from-scratch-with-python-step-by-step/)
- A clean **Streamlit interface** that acts as a whiteboard to draw digits
- Real-time predictions with **softmax-based probability scores**, visualized as a bar chart
- The entire project is lightweight, fast, and easy to understand — intended for learning and experimentation

---

## 🖼️ How It Works

1. You draw on a 280×280 canvas
2. The image is resized to 28×28, grayscale, and inverted
3. It’s normalized and flattened to match MNIST format
4. The input is passed into a manually-coded feedforward neural net
5. Prediction is returned, with a graph of the full probability distribution

---

## 🚧 Known Limitations

- The training dataset is **not the full MNIST database** — just a subset (~10k images), so it’s not highly robust
- This was **not designed to be production-grade** — more of a learning playground
- Some digits (like 7s or 9s) might be misclassified depending on how you draw them

---

## 🎯 Why I Built This

This project was part of my learning journey into:

- Understanding how **log loss**, **softmax**, and **backpropagation** really work
- Writing a feedforward neural net **without any libraries**
- Connecting that with an interactive UI using **Streamlit**

---

