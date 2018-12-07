"""
Testing model
Reference: https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
"""

import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import dataset

# CONSTANTS
IMG_SIZE = 20
NUM_CHANNELS = 3
TRAIN_PATH = 'TestingData'

# Variables
classes = os.listdir('TestingData')
images = []

# load testing data using path and file names
ch_names, ch_pixels, classes = dataset.load_data(TRAIN_PATH, classes)
#print(ch_pixels.shape)

# get the batch from loaded dataset
x_batch = ch_pixels
#    print(x_batch)

# get trained model
sess = tf.Session()
saver = tf.train.import_meta_graph('characters_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

# testing
graph = tf.get_default_graph()
y_pred = graph.get_tensor_by_name("y_pred:0")

x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name('y_true:0')
y_test_images = np.zeros((5564, 58)) # (0., 0., ... , 1.)

feed_dict_testing = {x: x_batch, y_true: y_test_images}
results=sess.run(y_pred, feed_dict=feed_dict_testing)

# Count total number of testing dataset and correct prediction to get
# accuracy of the model prediction.
count = 0
i = 0
for result in results:
    #print(classes[i])
    #print(result)

    trueIndex = classes[i] - 65
    print("Test set ", i,": ")
    if (result[trueIndex] == np.max(result)):
        count += 1
        print("True")
    else:
        print("False")
    i += 1

print("Correct prediction: ", count)
print("Total prediction: ", i)
print("Accuracy (%): ", count / i * 100)
