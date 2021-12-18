import math
from turtle import shape
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import imageio
from PIL import Image
from tensorflow.python.framework import ops
from cnn_utils import *
from tensorflow import keras
import os


base_path = os.path.abspath(".")

# 加载数据集
# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Example of a picture
# index = 6
# plt.imshow(X_train_orig[index])
# print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}

# 初始化参数
def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    gu = tf.initializers.GlorotUniform()
    
    ### START CODE HERE ### (approx. 2 lines of code)
    W1 = tf.Variable(name="W1", initial_value=gu(shape=[4, 4, 3, 8]))
    W2 = tf.Variable(name="W2", initial_value=gu(shape=[2, 2, 8, 16]))
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters

# parameters = initialize_parameters()
# print("W1 = " + str(parameters["W1"][1,1,1]))
# print("W2 = " + str(parameters["W2"][1,1,1]))

# 前向传播
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X -- 输入数据
    parameters -- 参数

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1'] # 用于卷积运算
    W2 = parameters['W2'] # 用于第二次卷积
    
    # 首先进行卷积运算
    Z1 = tf.nn.conv2d(X, W1, strides=1, padding='SAME')
    # 激活
    A1 = tf.nn.relu(Z1)
    # 池化
    P1 = tf.nn.max_pool2d(input=A1, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')
    # 第二次卷积
    Z2 = tf.nn.conv2d(P1, W2, strides=1, padding='SAME')
    # 激活
    A2 = tf.nn.relu(Z2)
    # 池化
    P2 = tf.nn.max_pool2d(A2, ksize=[1, 4, 4, 1], strides=[1,4,4,1], padding='SAME')
    # 转换为1维向量
    P2 = tf.keras.layers.Flatten()(P2)
    # 全联接
    Z3 = tf.keras.layers.Dense(6, activation='softmax')(P2)

    return Z3

def compute_cost(Z3, Y):
    """
    计算损失
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    return cost

# with tf.GradientTape() as tape:
#     np.random.seed(1)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(np.random.randn(4,64,64,3), parameters)
#     cost = compute_cost(Z3, np.random.randn(4,6))
#     print("Z3 = " + str(Z3))
#     print("cost = " + str(cost))
   
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.00007,
          num_epochs = 400, minibatch_size = 64, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    """
    seed = 3                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]                            
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()
    param2 = [parameters["W1"], parameters["W2"]]
    ### END CODE HERE ###
    
    num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
    
    for epoch in range(num_epochs):

        minibatch_cost = 0.
        
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            # IMPORTANT: The line that runs the graph on a minibatch.
            # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
            ### START CODE HERE ### (1 line)
            
            with tf.GradientTape() as tape:
                Z3 = forward_propagation(minibatch_X, parameters)
                cost = compute_cost(Z3, minibatch_Y)

            ### END CODE HERE ###
            optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
            gradients = tape.gradient(target=cost, sources=param2)
            _ = optimizer.apply_gradients(zip(gradients, param2))
            temp_cost = cost
            minibatch_cost += temp_cost / num_minibatches
            
        # Print the cost every epoch
        if print_cost == True and epoch % 5 == 0:
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if print_cost == True and epoch % 1 == 0:
            costs.append(minibatch_cost)
        
        
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # Calculate the correct predictions
    correct_prediction = tf.equal(tf.argmax(forward_propagation(X_train, parameters)), tf.argmax(Y_train))
    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    test_1 = tf.equal(tf.argmax(forward_propagation(X_test, parameters)), tf.argmax(Y_test))
    # Calculate accuracy on the test set
    test_accuracy = tf.reduce_mean(tf.cast(test_1, "float"))
    print("Train Accuracy:", accuracy)
    print("Test Accuracy:", test_accuracy)
                
    return accuracy, test_accuracy, parameters
    
_, _, parameters = model(X_train, Y_train, X_test, Y_test) 

fname = base_path+"/cnn/images/thumbs_up.jpg"
image = np.array(imageio.imread(fname))
my_image = np.array(Image.fromarray(image).resize((64,64))).reshape((64*64*3, 1))
plt.imshow(my_image)

def tf_model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
             num_epochs=100,minibatch_size=64,print_cost=True,isPlot=True):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(input_shape=(64,64,3),filters=32, kernel_size=3, padding='same',activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2, padding='same'))
    model.add(keras.layers.Conv2D(
                        filters=64, kernel_size=3, padding='same',
                        activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2,padding='same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(6, activation='softmax'))
    model.compile(optimizer='Adam',
            loss='mse',
            metrics=['accuracy'])
    model.summary()
    history = model.fit(x=X_train,y=Y_train,batch_size=64,epochs=num_epochs) 
    score = model.evaluate(x=X_test,y=Y_test) 
    model.save('model.h5') 
    return score,history
# score, history = tf_model(X_train, Y_train, X_test, Y_test, num_epochs=50, minibatch_size=64)
# print('测试集损失值:',str(score[0]*100)[:4]+'%') 
# print('测试集准确率:',str(score[1]*100)[:4]+'%') 
# plt.plot(np.squeeze(history.history['loss'])) 
# plt.ylabel('loss') 
# plt.xlabel('iterations (per tens)') 
# plt.show()

# from tensorflow.keras.models import load_model
# model = load_model(base_path+'/model.h5')
# a = X_test_orig[4]
# plt.imshow(a)
# print(model.predict(a.reshape(1,X_test_orig[1].shape[0],X_test_orig[1].shape[1], 3))[0].tolist().index(1))


