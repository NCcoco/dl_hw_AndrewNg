import sigmoid
import numpy as np

def iteration_param(dataset, Y, w1, w2, b, alpha): 
    j = 0; dw1 = 0; dw2 = 0; db = 0
    m = len(dataset)
    for i in range(0, m):
        z = dataset[i:i+1, 0] * w1 + dataset[i:i+1, 1] * w2 + b
        a = sigmoid(z)
        j += -(Y[i] * np.log(a) + (1 - Y[i]) * np.log(1 - a))
        dz = a - Y[i]

        dw1 += dataset[i:i+1, 0] * dz
        dw2 += dataset[i:i+1, 1] * dz
        db += dz

    j /= m
    dw1 /= m
    dw2 /= m
    db /= m

    w1 = w1 - alpha * dw1
    w2 = w2 - alpha * dw2
    b = b - alpha * db
    return w1,w2,b

