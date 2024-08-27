import nonlineardata as data
import math
import random

def softmax(predictions):
    m = max(predictions)
    temp = [math.exp(p - m) for p in predictions]
    total = sum(temp)
    return [t/total for t in temp]

def log_loss(activations, targets):
    losses = [-t * math.log(act) - (1 - t) * math.log(1 - act) for act, t in zip(activations, targets)]
    return sum(losses)

epochs = 1000
learning_rate = 0.5
input_count, hidden_count, output_count = 2, 8, 3

w_i_h = [[random.random() - 0.5 for _ in range(input_count)] for _ in range (hidden_count)]  #4 neurons, so 4 rows in weights column
#w_i_h = weights for inputs to hidden layers.

w_h_o = [[random.random() - 0.5 for _ in range(hidden_count)] for _ in range(output_count)]
#w_h_o = weights from hidden layers to output layers

b_i_h = [0 for _ in range(hidden_count)]  #4 hidden neurons
b_h_o = [0 for _ in range(output_count)]    #3 output neurons

for epoch in range(epochs):
    pred_h = [[sum([w * a for w, a in zip(weights, inp)]) + bias for weights, bias in zip(w_i_h, b_i_h)] for inp in data.inputs]
    #print(len(pred_h))  #dimensions of pred_h, shd be 60
    #print(len(pred_h[0]))   #shd be 4 (hidden layer) neurons

    act_h = [[max(0, p) for p in pred] for pred in pred_h]  #apply ReLU
    #print(act_h)
    pred_o = [[sum([w * a for w, a in zip(weights, inp)]) + bias for weights, bias in zip(w_h_o, b_h_o)] for inp in act_h]
    act_o = [softmax(predictions) for predictions in pred_o]
    #print(act_o)
    cost = sum([log_loss(a, t) for a, t in zip(act_o, data.targets)]) / len(act_o)
    print(f"epoch: {epoch}  cost:{cost:.4f}")

    # error derivative

    errors_d_o = [[a - t for a, t in zip(ac, ta)] for ac, ta in zip(act_o, data.targets)]
    w_h_o_transpose = list(zip(*w_h_o))
    errors_d_h = [[sum([d * w for d, w in zip(deltas, weights)]) * (0 if p<=0 else 1) for weights, p in zip(w_h_o_transpose, pred)] for deltas, pred in zip(errors_d_o, pred_h)]
    # gradient hidden ---> output
    act_h_transpose = list(zip(*act_h))
    errors_d_o_transpose = list(zip(*errors_d_o))
    w_h_o_d = [[sum([d * a for d, a in zip(deltas, act)]) for deltas in errors_d_o_transpose] for act in act_h_transpose]
    b_h_o_d = [sum([d for d in deltas]) for deltas in errors_d_o_transpose]

    #gradient input--->hidden
    inputs_transpose = list(zip(*data.inputs))
    errors_d_h_transpose = list(zip(*errors_d_h))
    w_i_h_d = [[sum([d * a for d, a in zip(deltas, act)]) for deltas in errors_d_h_transpose] for act in inputs_transpose]
    b_i_h_d = [sum([d for d in deltas]) for deltas in errors_d_h_transpose]

    #update weights and biases for all layers:
    w_h_o_d_transpose = list(zip(*w_h_o_d))
    for y in range(output_count):
        for x in range(hidden_count):
            w_h_o[y][x] -= learning_rate * w_h_o_d_transpose[y][x] / len(data.inputs)
        b_h_o[y] -= learning_rate * b_h_o_d[y]/len(data.inputs)
    
    w_i_h_d_transpose = list(zip(*w_i_h_d))
    for y in range(hidden_count):
        for x in range(input_count):
            w_i_h[y][x] -= learning_rate * w_i_h_d_transpose[y][x] / len(data.inputs)
        b_i_h[y] -= learning_rate * b_i_h_d[y] / len(data.inputs)

# test the network:

pred_h = [[sum([w * a for w, a in zip(weights, inp)]) + bias for weights, bias in zip(w_i_h, b_i_h)] for inp in data.test_inputs]
act_h = [[max(0, p) for p in pre] for pre in pred_h] 
pred_o = [[sum([w * a for w, a in zip(weights, inp)]) + bias for weights, bias in zip(w_h_o, b_h_o)] for inp in act_h]
act_o = [softmax(predictions) for predictions in pred_o]
correct = 0
for a, t in zip(act_o, data.test_targets):
    if a.index(max(a)) == t.index(max(t)):
        correct+=1

print(f"correct: {correct}/{len(act_o)} ({correct/len(act_o):%})")