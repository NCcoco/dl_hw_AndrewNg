import math
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
import scipy
from PIL import Image
import skimage.transform as sktrans
import os
import imageio

base_path = os.path.abspath(".")
print(base_path)
# 线性模型
def linear_function():
    np.random.seed(1)
    
    X = tf.constant(np.random.randn(3,1), name="X")
    W = tf.constant(np.random.randn(4,3), name="Y")
    b = tf.constant(np.random.randn(4,1), name="b")
    Y = tf.matmul(W,X) + b
    
    return Y
    
# print(linear_function())
    
def sigmoid(z):
    # 将普通变量转换为tf变量
    z = tf.Variable(z, dtype=float)
    return tf.sigmoid(z)

# print("sigmoid(0) = " + str(sigmoid([0, 12, 99, -5])))

def cost(logists, labels):
    
    ### START CODE HERE ### 
    # Create the placeholders for "logits" (z) and "labels" (y) (approx. 2 lines)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logists=logists, labels=labels)
    ### END CODE HERE ###
    return cost


# logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
# y = tf.Variable(np.array([0,0,1,1]), dtype='float')
# cost = cost(logits, y)
# print ("cost = " + str(cost))

def one_hot_matrix(labels, C):
    
    ### START CODE HERE ###
    
    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(value = C, name="C")
    
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot = tf.one_hot(labels, C, axis=0)
    
    
    return one_hot

# labels = np.array([1,2,3,0,2,1])
# labels = tf.Variable(labels)
# one_hot = one_hot_matrix(labels, C = 4)
# print ("one_hot = " + str(one_hot))

def ones(shape):
    ones = tf.ones(shape)
    return ones
    
# print(str(ones([3,2])))

### 构建一个自己的神经网络！

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# index = 0
# plt.imshow(X_train_orig[index])
# plt.show()
# print("y = " + str(np.squeeze(Y_train_orig[:, index])))

# print(X_train_orig.shape)
# print(Y_train_orig.shape)

X_train = X_train_orig.reshape((X_train_orig.shape[0], -1)).T
X_test = X_test_orig.reshape((X_test_orig.shape[0], -1)).T
# 数值归一：
X_train = X_train / 255.0
X_test = X_test / 255.0

Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

# print ("number of training examples = " + str(X_train.shape[1]))
# print ("number of test examples = " + str(X_test.shape[1]))
# print ("X_train shape: " + str(X_train.shape))
# print ("Y_train shape: " + str(Y_train.shape))
# print ("X_test shape: " + str(X_test.shape))
# print ("Y_test shape: " + str(Y_test.shape))

# 成功将
def initialize_parameters():
    gu = tf.initializers.GlorotUniform(seed=1)
    zeros_init = tf.zeros_initializer()
    W1 = tf.Variable(name='W1', initial_value=gu(shape=[25, 12288]))
    b1 = tf.Variable(name='b1', initial_value=zeros_init(shape=[25, 1]))
    W2 = tf.Variable(name='W2', initial_value=gu(shape=[12, 25]))
    b2 = tf.Variable(name='b2', initial_value=zeros_init(shape=[12, 1]))
    W3 = tf.Variable(name='W3', initial_value=gu(shape=[6, 12]))
    b3 = tf.Variable(name='b3', initial_value=zeros_init(shape=[6, 1]))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

# parameters = initialize_parameters()
# print("W1 = " + str(parameters["W1"].shape))
# print("b1 = " + str(parameters["b1"].shape))
# print("W2 = " + str(parameters["W2"].shape))
# print("b2 = " + str(parameters["b2"].shape))


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    ### START CODE HERE ### (approx. 5 lines)
    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3
    ### END CODE HERE ###
    
    return Z3


# parameters = initialize_parameters()
# Z3 = forward_propagation(X_train, parameters)
# print("Z3 = " + str(Z3))

# GRADED FUNCTION: compute_cost 

def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    ### END CODE HERE ###
    
    return cost

# parameters = initialize_parameters()
# Z3 = forward_propagation(X_train, parameters)
# cost = compute_cost(Z3, Y_train)
# print("Z3 = " + str(Z3))
# print(cost)

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.00014,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters() 
    ### END CODE HERE ###
    
    # Do the training loop
    for epoch in range(num_epochs):
        epoch_cost = 0.                           # Defines a cost related to an epoch
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            ## 把我们强大的自动微分加进来：
            with tf.GradientTape() as tape:
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                Z3 = forward_propagation(minibatch_X, parameters)
                cost = compute_cost(Z3, minibatch_Y)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            _ = optimizer.minimize(cost, var_list=parameters, tape=tape)
            minibatch_cost = cost
            ### END CODE HERE ###
            epoch_cost += minibatch_cost / num_minibatches
        # Print the cost every epoch
        if print_cost == True and epoch % 100 == 0:
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost == True and epoch % 5 == 0:
            costs.append(epoch_cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # lets save the parameters in a variable
    print ("Parameters have been trained!")

    # Calculate the correct predictions
    correct_prediction = tf.equal(tf.argmax(forward_propagation(X_train, parameters)), tf.argmax(Y_train))
    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    test_1 = tf.equal(tf.argmax(forward_propagation(X_test, parameters)), tf.argmax(Y_test))
    # Calculate accuracy on the test set
    test_accuracy = tf.reduce_mean(tf.cast(test_1, "float"))

    print ("Train Accuracy:", accuracy)
    print ("Test Accuracy:", test_accuracy)
    
    return parameters

parameters = model(X_train, Y_train, X_test, Y_test)




## START CODE HERE ## (PUT YOUR IMAGE NAME) 
my_image = "thumbs_up.jpg"
## END CODE HERE ##

# We preprocess your image to fit your algorithm.
fname = base_path + "/tutorial/images/" + my_image
image = np.array(imageio.imread(fname))
my_image = np.array(sktrans.resize(image, (64,64,3))).reshape((64*64*3, 1))
my_image_prediction = predict(my_image, parameters)

plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))




























