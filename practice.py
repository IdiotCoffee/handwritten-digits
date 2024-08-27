#training data:
inputs = [1, 3, 6, 7, 8, 9]
targets = [4, 13, 28, 32, 35]

w = 0.1
learning_rate = 0.01
b = 0.1
epochs = 100000

def predict(i):
    return w * i + b

for _ in range(epochs):
    predicted_vals = [predict(i) for i in inputs]
    err_vals = [(t - p)**2 for t, p in zip(predicted_vals, targets)]
    cost = sum(err_vals)/len(targets)
    print(f"weight: {w:.2f}     bias: {b:.2f}   cost: {cost:.2f}")

    err_der = [2 * (p - t) for p, t in zip(predicted_vals, targets)]
    w_delta = [(e * i) for e, i in zip(err_der, inputs)]
    b_delta = [(e * 1) for e in err_der]

    avg_w = sum(w_delta)/len(w_delta)
    avg_b = sum(b_delta)/len(b_delta)

    w -= learning_rate * avg_w
    b -= learning_rate * avg_b


#test inputs
test_inputs = [5, 10, 11, 12]
test_target = [21, 42, 33, 50]
pred = [predict(i) for i in test_inputs]
for i, t, p in zip(test_inputs, test_target, pred):
    print(f"Input: {i}  Targer: {t} Predicted: {p:.4f}")
