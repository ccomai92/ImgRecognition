"""
Training model
Reference: https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
"""

import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Adding seed so that random initializaiton is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# Prepare input data
classes = os.listdir('Dataset')

# CONSTANTS
VALIDATION_SIZE = 0.1   # 10 folds validation
NUM_CHANNELS = 3        # RGB images are 3D image matrices
TRAIN_PATH = 'Dataset'  # Folder name for input datasets
IMG_SIZE = 20           # Size of input data 20 * 20
BATCH_SIZE = 32         # Size of one batch

# Reading input datasets (training and validation)
data = dataset.read_train_sets(TRAIN_PATH, classes, VALIDATION_SIZE)

# Log data for training
print("Complete Reading input data. ")
print("Number of images in training-set:\t{}".format(len(data.train.img_labels())))
print("Number of labels in validation-set:\t{}".format(len(data.valid.img_labels())))


session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS], name='x')

## labels
# unique font type (no duplicates)
num_classes = data.train.img_labels()[0].size

# storing predictions
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

## Network graph params for convolution layers

# 3*3 filter size with 32 different filters
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

# 3*3 filter size with 64 different filters
filter_size_conv3 = 3
num_filters_conv3 = 64

# Fully connected layer
fc_layer_size = 128

def create_weights(shape):
    # Randomly initializing weights
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    # Randomly initializing biases
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input, num_input_channels,
                            conv_filter_size, num_filters):
    # initializing weights
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])

    # initializing biases
    biases = create_biases(num_filters)

    # Defining 2D convolutional layer applying weights and padding.
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

    # Add biases on the filtered images.
    layer += biases

    # max-pooling from filtered images
    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Output of pooling is fed to Relu
    layer = tf.nn.relu(layer)

    return layer

def create_flatten_layer(layer):
    # shape of the layer [batch_size, img_size, img_size, num_channels]
    layer_shape = layer.get_shape()

    ## num of features will be img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()

    ## flatten the layer
    layer = tf.reshape(layer, [-1, num_features])

    return layer

# fully connected layer
def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    # define trainable weights and biases
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # fc layer takes input x and produce wx + b
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


# Flow of tensor
layer_conv1 = create_convolutional_layer(input=x,
                num_input_channels=NUM_CHANNELS,
                conv_filter_size=filter_size_conv1,
                num_filters=num_filters_conv1)

layer_conv2 = create_convolutional_layer(input=layer_conv1,
                num_input_channels=num_filters_conv1,
                conv_filter_size=filter_size_conv2,
                num_filters=num_filters_conv2)

layer_conv3 = create_convolutional_layer(input=layer_conv2,
                num_input_channels=num_filters_conv2,
                conv_filter_size=filter_size_conv3,
                num_filters=num_filters_conv3)

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                    num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                    num_outputs=fc_layer_size,
                    use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                    num_inputs=fc_layer_size,
                    num_outputs=num_classes,
                    use_relu=False)

# contains the predicted probability of each class for
# each input image
y_pred = tf.nn.softmax(layer_fc2, name='y_pred')
y_pred_cls = tf.argmax(y_pred, dimension=1)

# cross_entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
                                                        labels=y_true)
# average cross_entropy
cost = tf.reduce_mean(cross_entropy)


# run optimizer operation inside session.run(), inorder to calculate
# the whole network will have to be run and we will pass the training image
# in a feed_dict
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())

# y_true_batch is BATCH_SIZE of labels
# x_batch is BATCH_SIZE of images
def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0
saver=tf.train.Saver()
def train(num_iterations):
    # Actual training for the model
    global total_iterations

    for i in range(total_iterations, total_iterations + num_iterations):

        y_true_batch, x_batch, cls_batch = data.train.next_batch(BATCH_SIZE)
        y_valid_batch, x_valid_batch, valid_cls_batch = data.valid.next_batch(BATCH_SIZE)

        feed_dict_tr = {x: x_batch, y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples() / BATCH_SIZE) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples() / BATCH_SIZE))
            epochs.append(epoch)
            validation_loss.append(val_loss)
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, './characters_model')

    total_iterations += num_iterations


validation_loss = []
epochs = []
train(num_iterations=40000)

validation_loss = np.array(validation_loss)
epochs = np.array(epochs)

plt.plot(epochs, validation_loss)
plt.show()
#plt.savefig('Validation_Loss.png')