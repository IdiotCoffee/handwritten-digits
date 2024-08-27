inputs = [1, 2, 3, 4]
targets = [12, 14, 16, 18]

b = 0.3
w = 0.1
learning_rate = 0.1
epochs = 100

def predict(i):
    return w * i +b

#training the network:
#weight and slope are the same thing. thats why its denoted by w
#every iteration below, in the for loop, is called an epoch.
for _ in range(epochs):
    pred = [predict(i) for i in inputs]
    errors = [(t - p)**2 for p, t in zip(pred, targets)] #using mean squared error, which gives overflow error in o/p
    cost = sum(errors)/len(targets)
    print(f"Weight: {w:.2f}     Bias: {b:.2f}     Cost: {cost:.2f}")


    #calculating the slope, i.e, the error_derivative at all points of error:
    err_d = [2*(p-t) for p, t in zip(pred, targets)]
    delta = [e*i for e, i in zip(err_d, inputs)]
    b_delta = [e*1 for e in err_d]

    avg_b = sum(b_delta)/len(b_delta)
    avg_d = sum(delta)/len(delta)
    #delta is the changes that each training sample wants to make to the weight
    w -= learning_rate * avg_d
    b-= learning_rate * avg_b



#test inputs
test_inputs = [5, 6, 7, 8]
test_target = [20, 22, 24, 26]
pred = [predict(i) for i in test_inputs]
for i, t, p in zip(test_inputs, test_target, pred):
    print(f"Input: {i}  Targer: {t} Predicted: {p:.4f}")