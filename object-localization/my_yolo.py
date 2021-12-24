import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import pandas as pd
import PIL
import imageio
import yolo_utils
from keras_darknet19 import *
import keras_yolo
from utils import *

tf.compat.v1.disable_eager_execution()
base_path = os.path.abspath(".") + "/"
# yolo_utils_path = base_path + "object-localization/Car detection for Autonomous Driving/yolo_utils.py"
# import importlib.util
# spec = importlib.util.spec_from_file_location("yolo_utils", yolo_utils_path)
# yolo_utils = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(yolo_utils)
# yolo_utils.MyClass()

"""
相当于yolo已经提供了如何从训练数据集中确定预测框、锚框的大小，位置
"""
# yolo过滤器
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = float(0.6)):
    """
    Args:
        box_confidence ([type]): 置信度   (19×19,5,1) 
        boxes ([type]): 19x19x5 个锚框对象   (19×19,5,4) 
        box_class_probs ([type]): 每个锚框的概率 (19×19,5,80) 
        threshold (float, optional): 去除IoU小于此值的预测框

    Returns:
        [type]: [description]
    """
    # 计算boxes的分值 按我理解， 用最后一个维度乘以box_class_probs的最后一个维度，但是前面的维度又必须一一对应
    box_scores = box_confidence * box_class_probs
    # 找出最大值的下标，并记录最大值的值,
    box_classes = keras.backend.argmax(box_scores, axis=-1)
    box_class_scores = keras.backend.max(box_scores, axis=-1, keepdims=False)
    box_class_scores = tf.cast(box_class_scores, dtype='float32')

    # 第三步： 创建蒙版
    filter_mask = (box_class_scores > threshold)
    
    # 第四步：将蒙版应用到分数，boxes、分类中
    scores = tf.boolean_mask(box_class_scores, filter_mask)
    boxes = tf.boolean_mask(boxes, filter_mask)
    classes = tf.boolean_mask(box_classes, filter_mask)
    
    return scores, boxes, classes

# box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
# boxes = tf.random.normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
# box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
# print(box_confidence.shape, boxes.shape, box_class_probs.shape)
# scores, boxes, classes = yolo_filter_boxes(box_confidence=box_confidence, boxes=boxes, box_class_probs=box_class_probs)

# print("scores[2] = " + str(scores[2].numpy()))
# print("boxes[2] = " + str(boxes[2].numpy()))
# print("classes[2] = " + str(classes[2].numpy()))
# print("scores.shape = " + str(scores.shape))
# print("boxes.shape = " + str(boxes.shape))
# print("classes.shape = " + str(classes.shape))

def iou(box1, box2):
    
    # 计算box1和box2交点的(y1,x1,y2,x2)坐标。 计算其面积。
    # 获取两个框的重叠部分
    xi1 = max(box1[0], box2[0]) 
    yi1 = max(box1[1], box2[1]) # 对比两个左上角，取最大值得到共同部分的左上角
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3]) # 对比两个右下角，取最小值得到共同部分的右下角
    # 计算面积
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    
    # 计算box1与box2的联合面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # 计算iou
    iou = float(inter_area) / float(union_area)
    return iou

# box1 = (2, 1, 4, 3)
# box2 = (1, 2, 3, 4) 
# print("iou = " + str(iou(box1, box2)))

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = float(0.5)):
    """
    运用非最大抑制过滤预测框
    
    Arguments:
    scores -- 分数
    boxes -- 预测框
    classes -- 分类
    max_boxes -- 最大框数
    iou_threshold -- 过滤iou高于此值的框
    
    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    """
    max_box_tensor = keras.backend.variable(max_boxes, dtype='int32')
    tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.variables_initializer([max_box_tensor]))
    # 使用tensorflow的非最大抑制得到想要的框
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_box_tensor, iou_threshold, name=None)
    
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    
    return scores, boxes, classes
    
# scores = tf.random.normal([54,], mean=1, stddev=4, seed = 1)
# boxes = tf.random.normal([54, 4], mean=1, stddev=4, seed = 1)
# classes = tf.random.normal([54,], mean=1, stddev=4, seed = 1)
# scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
# print("scores[2] = " + str(scores[2].numpy()))
# print("boxes[2] = " + str(boxes[2].numpy()))
# print("classes[2] = " + str(classes[2].numpy()))
# print("scores.shape = " + str(scores.numpy().shape))
# print("boxes.shape = " + str(boxes.numpy().shape))
# print("classes.shape = " + str(classes.numpy().shape))

# ?
def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, scores_threshold=0.6, iou_threshold=0.5):
    
    # 将yolo输出中的信息单独提取出来
    boxes_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    # 将基于中心点和宽高表示的数据转换为左上角和右下角表示的数据
    boxes = keras_yolo.yolo_boxes_to_corners(box_xy, box_wh)
    
    # 使用分数抑制过滤不需要的框
    scores, boxes, classes = yolo_filter_boxes(boxes_confidence, boxes, box_class_probs, scores_threshold)
    # 将原本不符合格式的框缩放为符合image_shape的框
    boxes = yolo_utils.scale_boxes(boxes, image_shape) 
    
    # 使用非极大值抑制
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
    
    # 将输入的数据，转换为
    return scores, boxes, classes

yolo_outputs = (tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                    tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))

# scores, boxes, classes = yolo_eval(yolo_outputs)
# print("scores[2] = " + str(scores[2].numpy()))
# print("boxes[2] = " + str(boxes[2].numpy()))
# print("classes[2] = " + str(classes[2].numpy()))
# print("scores.shape = " + str(scores.numpy().shape))
# print("boxes.shape = " + str(boxes.numpy().shape))
# print("classes.shape = " + str(classes.numpy().shape))



def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.
    
    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """

    # Preprocess your image
    image, image_data = yolo_utils.preprocess_image(
        base_path + "object-localization/Car detection for Autonomous Driving/images/" + image_file, model_image_size = (608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    ### START CODE HERE ### (≈ 1 line)
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input:image_data, K.learning_phase():0.})
    ### END CODE HERE ###

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = yolo_utils.generate_colors(class_names)
    # Draw bounding boxes on the image file
    yolo_utils.draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join(base_path + "object-localization/Car detection for Autonomous Driving/out", image_file), quality=90)
    # Display the results in the notebook
    output_image = imageio.imread(os.path.join(base_path + "object-localization/Car detection for Autonomous Driving/out", image_file))
    plt.imshow(output_image)
    plt.show()
    
    return out_scores, out_boxes, out_classes

with tf.compat.v1.Session() as sess:
    # 在图片上测试预处理模型
    class_names = yolo_utils.read_classes(base_path + "object-localization/Car detection for Autonomous Driving/model_data/coco_classes.txt")
    anchors = yolo_utils.read_anchors(base_path + "object-localization/Car detection for Autonomous Driving/model_data/yolo_anchors.txt")
    image_shape = (720., 1280.)
    yolo_model = models.load_model(base_path+"object-localization/yolo.h5")
    yolo_model.summary()
    yolo_outputs = keras_yolo.yolo_head( yolo_model.output, anchors, len(class_names))
    print()
    scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
    # print("scores.shape = " + str(scores.numpy().shape))
    # print("boxes.shape = " + str(boxes.numpy().shape))
    # print("classes.shape = " + str(classes.numpy().shape))
    out_scores, out_boxes, out_classes = predict(sess, "0008.jpg")








