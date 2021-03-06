# 正则化优化的神经网络
# 思考如下问题： 
# 1.如何确定是偏差还是方差
# 2.选择哪种正则化优化方法 L2， L1，or dropout
import numpy as np
import matplotlib.pyplot as plt
from reg_utils import *
from testCases import *
import sklearn
import sklearn.datasets
import scipy.io


plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

### 法国足球公司需要进行守门员发球位置的建议
# 现在，为我提供了10场足球赛的历史数据
train_x, train_y, test_x, test_y = load_2D_dataset()

## 首先尝试没有正则化的模型进行预测

def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    learning_rate -- learning rate of the optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.
    
    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """
        
    grads = {}
    costs = []                            # to keep track of the cost
    m = X.shape[1]                        # number of examples
    layers_dims = [X.shape[0], 20, 3, 1]
    
    # Initialize parameters dictionary.
    parameters = initialize_parameters(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        
        # Cost function
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
            
        # Backward propagation.
        assert(lambd==0 or keep_prob==1)    # it is possible to use both L2 regularization and dropout, 
                                            # but this assignment will only explore one at a time
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
    
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

# 在没有正则化的模型下进行训练和预测
# parameters = model(train_x, train_y)
# print("在训练集下的表现：")
# predict_train = predict(train_x, train_y, parameters)
# print("在测试集下的表现：")
# predict_test = predict(test_x, test_y, parameters)

# plt.title("model with regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75, 0.40])
# axes.set_ylim([-0.75, 0.65])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_x, train_y)

## 上面的决策边界可以看出确实有些过拟合了。

# 下面使用L2范数作为正则项（regularization）
def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula (2) above.
    
    Arguments:
    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model
    
    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost
    
    L2_regularization_cost = lambd/(2*m) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost

# A3, Y_assess, parameters = compute_cost_with_regularization_test_case()
# print("损失为：" + str(compute_cost_with_regularization(A3, Y_assess, parameters, 0.1)))

# 定义反向传播正则化函数
def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    Implements the backward propagation of our baseline model to which we added an L2 regularization.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularization hyperparameter, scalar
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    ###
    dW3 = 1./m * np.dot(dZ3, A2.T) + np.multiply(lambd/m, W3)
    ###
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T, dZ3)
    
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    ###
    dW2 = 1./m * np.dot(dZ2, A1.T) + np.multiply(lambd/m, W2)
    ###
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    ###
    dW1 = 1./m * np.dot(dZ1, X.T) + np.multiply(lambd/m, W1)
    ###
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

# X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()
# gradients = backward_propagation_with_regularization(X_assess, Y_assess, cache, 0.7)
# print ("dW1 = "+ str(gradients["dW1"]))
# print ("dW2 = "+ str(gradients["dW2"]))
# print ("dW3 = "+ str(gradients["dW3"]))

# parameters = model(train_x, train_y, lambd = 0.7)
# print ("在训练集上的表现:")
# predictions_train = predict(train_x, train_y, parameters)
# print ("在测试集上的表现:")
# predictions_test = predict(test_x, test_y, parameters)


## 可以看出正则化确实实现了去除过拟合问题
# plt.title("Model with L2-regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_x, train_y)


### 前向传播
def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    
    Arguments:
    X -- input dataset, of shape (2, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    A3 -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """
    
    np.random.seed(1)
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    
    ## 得到随机矩阵D1
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = np.int64(D1 < keep_prob)
    A1 = np.multiply(A1, D1)
    A1 = A1 / keep_prob
    
    # 
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = np.int64(D2 < keep_prob)
    A2 = np.multiply(A2, D2)
    A2 = A2 / keep_prob
    
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache


# X_assess, parameters = forward_propagation_with_dropout_test_case()

# A3, cache = forward_propagation_with_dropout(X_assess, parameters, keep_prob = 0.7)
# print ("A3 = " + str(A3))


# 带有dropout的反向传播
def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    Implements the backward propagation of our baseline model to which we added dropout.
    
    Arguments:
    X -- input dataset, of shape (2, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation_with_dropout()
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)

    # 关闭神经元并且同步缩放dA2
    dA2 = np.multiply(dA2, D2)
    dA2 = dA2 / keep_prob
    
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    
    # 关闭神经元并同步缩放dA1
    dA1 = np.multiply(dA1, D1)
    dA1 = dA1 / keep_prob
    
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
    
# X_assess, Y_assess, cache = backward_propagation_with_dropout_test_case()

# gradients = backward_propagation_with_dropout(X_assess, Y_assess, cache, keep_prob = 0.8)

# print ("dA1 = " + str(gradients["dA1"]))
# print ("dA2 = " + str(gradients["dA2"]))
    
parameters = model(train_x, train_y, keep_prob = 0.86, learning_rate = 0.3)

print ("On the train set:")
predictions_train = predict(train_x, train_y, parameters)
print ("On the test set:")
predictions_test = predict(test_x, test_y, parameters)
    
plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_x, train_y)
