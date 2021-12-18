from re import S
from unicodedata import name
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, utils, initializers
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

from resnets_utils import *

import os
base_path = os.path.abspath(".") + "/Resnet/"


# 恒等模块
def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- 输入图像 (m, n_H_prev, n_W_prev, n_C_prev)
    f -- 滤波器大小
    filters -- 滤波器个数
    stage -- 用于确定第几层的整数
    block -- 用于命名层的字符串
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        name=conv_name_base + '2a',
        kernel_initializer=initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = layers.Activation('relu')(X)

    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = layers.Conv2D(
        filters=F2,
        kernel_size=(f, f),
        strides=(1, 1),
        padding='same',
        name=conv_name_base + '2b',
        kernel_initializer=initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)
    # Third component of main path (≈2 lines)
    X = layers.Conv2D(
        filters=F3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        name=conv_name_base + '2c',
        kernel_initializer=initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('ReLU')(X)
    ### END CODE HERE ###

    return X


# np.random.seed(1)
# X = np.random.randn(3, 4, 4, 6)
# out = identity_block(X, f = 2, filters=[2,4,6], stage=1, block='a')
# print(out.shape)
# print("out = " + str(out[1][1][0]))


# 卷积块
def convolutional_block(X, f, filters, stage, block, s=1):
    """
    实现卷积块

    Args:
        X ([type]): 输入值
        f ([type]): 滤波器大小
        filters: 每层的过滤器个数
        stage ([type]): 用于确定第几层的整数
        block ([type]): 用于命名层的字符串

    Returns:
        [type]: [description]
    """
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 获取滤波器个数
    (F1, F2, F3) = filters

    # 记录当前值
    X_shortcut = X
    
    # 主路径的第一部分 （一个卷积层，一个batch归一，一个激活）
    X = layers.Conv2D(filters=F1,
                      kernel_size=(1, 1),
                      # padding='valid',
                      strides=(s, s),
                      name=conv_name_base + '2a')(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = layers.Activation('relu')(X)

    # 主路径的第二部分 （一个卷积层，一个batch归一，一个激活）
    X = layers.Conv2D(filters=F2,
                      kernel_size=(f, f),
                      strides=(1, 1),
                      padding='same',
                      name=conv_name_base + '2b')(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)

    # 主路径的第三部分 （一个卷积层，一个batch归一）
    X = layers.Conv2D(filters=F3,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='valid',
                      name=conv_name_base + '2c')(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # 以上计算后，X的shape为 (w/s) - f + 1
    # 压缩X_shortcut的大小
    X_shortcut = layers.Conv2D(
        filters=F3,
        kernel_size=(1, 1),
        strides=(s, s),
        name=conv_name_base + '1a',
        kernel_initializer=initializers.glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = layers.BatchNormalization(axis=3,
                                           name=bn_name_base + '1')(X_shortcut)

    # 捷径 （一个过去值，一个激活）
    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)

    return X


# np.random.seed(1)
# X = np.random.randn(3, 4, 4, 6)
# out = convolutional_block(X, f=2, filters = [2, 4, 6], stage = 1, block = 'a', s=2)
# print(out.shape)
# print("out = " + str(out[0][1][1][0]))


# 50层的残差网络
def ResNet50(input_shape, classes=6):
    """
    流行的 ResNet50 实现了以下架构：
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- 输入值的形状
    classes -- 分类的类型数目

    Returns:
    model -- a Model() instance in Keras
    """
    # 定义输入数据的shape
    X_input = layers.Input(input_shape)

    # 0填充 将变为 64+3x2 = 70
    X = layers.ZeroPadding2D((3, 3))(X_input)
    print(X.shape)
    # Stage 1 (将变为（70-7）/2 + 1 = 32）
    X = layers.Conv2D(
        64, (7, 7),
        strides=(2, 2),
        name='conv1',
        kernel_initializer=initializers.glorot_uniform(seed=0))(X)
    print(X.shape)
    X = layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = layers.Activation('relu')(X)
    # （将变为32 - 3 / 2 + 1 = 15）
    X = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)
    print(X.shape)
    # Stage 2
    X = convolutional_block(X,
                            f=3,
                            filters=[64, 64, 256],
                            stage=2,
                            block='a',
                            s=1)
    print(X.shape)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    print(X.shape)
    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X,
                            f=3,
                            filters=[128, 128, 512],
                            stage=3,
                            block='a',
                            s=2)
    print("第三阶段：",X.shape) 
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='b')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='c')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X,
                            f=3,
                            filters=[256, 256, 1024],
                            stage=4,
                            block='a',
                            s=2)
    print("第四阶段：",X.shape) 
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='b')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='c')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='d')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='e')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='f')
    print(X.shape)
    # Stage 5 (≈3 lines)
    X = convolutional_block(X,
                            f=3,
                            filters=[512, 512, 2048],
                            stage=5,
                            block='a',
                            s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='b')
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = layers.AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

    ### END CODE HERE ###

    # output layer
    X = layers.Flatten()(X)
    X = layers.Dense(classes,
                     activation='softmax',
                     name='fc' + str(classes),
                     kernel_initializer=initializers.glorot_uniform(seed=0))(X)

    # Create model
    model = models.Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

# model = ResNet50(input_shape=X_train.shape[1:4], classes=6)
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])



# print("number of training examples = " + str(X_train.shape[0]))
# print("number of test examples = " + str(X_test.shape[0]))
# print("X_train shape: " + str(X_train.shape))
# print("Y_train shape: " + str(Y_train.shape))
# print("X_test shape: " + str(X_test.shape))
# print("Y_test shape: " + str(Y_test.shape))

# model.fit(X_train, Y_train, epochs=10, batch_size=32)
# preds = model.evaluate(X_test, Y_test)
# model.save('Resnet/ResNet50.h5') 
# print("Loss = " + str(preds[0]))
# print("Test Accuracy = " + str(preds[1]))

import imageio  
from PIL import Image  
model = models.load_model(base_path + 'ResNet50.h5')  
img_path = base_path + 'images/my_image.jpg'  

img = image.load_img(img_path, target_size=(64, 64))  
x = image.img_to_array(img)  
x = np.expand_dims(x, axis=0)  
x = preprocess_input(x)  


# my_image = np.array(imageio.imread(img_path))  
# my_image = np.array(Image.fromarray(my_image).resize((64,64))).reshape((64*64*3, 1))  

## 错误率极高
print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")  
np.set_printoptions(suppress=True)
print(model.predict(X_test)) 

index = 0
for i in X_test:
    plt.title(str(index))
    plt.imshow(i)
    plt.show()
    index += 1

