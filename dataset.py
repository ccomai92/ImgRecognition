"""
Class to read the input data.
reads cvs file from provided training and testing data folders.
Reference: https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
"""

import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


class DataSet(object):
    """
        DataSet class that holds the dataset for training, validating.
        Initial input:
            - img_labels
            - 4D list of images(3D)
            - corresponding class in 1D list
        self._num_examples: the integer number of examples
        self._img_labels: the 2D list of unique labels where corresponding index label holds 1.0,
                            other wise values are 0.0
        self._images: 4D list contains number of examples of 20 * 20 * 3 image matrices
        self._cls: the list of image classes in 1D list
        self._epochs_done: number of epochs done in the iterations
        self._index_in_epoch: based on the batch size and the number of iterations, it holds the index
                            of data in current epoch.
    """
    def __init__(self, img_labels, images, classes):
        self._num_examples = img_labels.shape[0]
        self._img_labels = img_labels
        self._images = images

        self._cls = classes
        self._epochs_done = 0
        self._index_in_epoch = 0

    def num_examples(self):
        return self._num_examples

    def img_labels(self):
        return self._img_labels

    def images(self):
        return self._images

    def epochs_done(self):
        return self._epochs_done

    def cls(self):
        return self._cls

    def next_batch(self, batch_size):
        """Return the next 'batch_size' examples from this data set"""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._img_labels[start:end], self._images[start:end], self._cls[start:end]




def load_data(train_path, fonts):
    """
    Input: data path and the list of font names.
    Output:
        - ch_names: list of unique labels holding 0.0 or 1.0
        - ch_pixels: 4D list containing N different 3D RGB image matrices
        - classes: name of class (character label) index corresponding
                    image in ch_pixels
    """
    ch_pixels = []
    ch_names = []
    classes = []

    # print(fonts)
    for font in fonts:
        path = os.path.join(train_path, font)
        data_frame = pd.read_csv(path)
        data = data_frame.values    # table

        for d in data:
            # character in integer
            if (d[2] >= 65 and d[2] <= 122):
                classes.append(d[2])

                # pixels in 20 * 20 * 3
                pixels = d[-400:].reshape(20, 20)
                result = np.zeros((20, 20, 3), dtype=int)
                temp = pixels[:, :]

                r = temp
                g = temp
                b = temp
                result[:, :, 0] = r
                result[:, :, 1] = g
                result[:, :, 2] = b

                ch_pixels.append(result)

    unique_classes = np.unique(classes)
    for character_int in classes:
        temp = np.zeros(unique_classes.size)
        #print(character_int)
        index = character_int - 65
        temp[index] = 1.0
        #print(temp)
        ch_names.append(temp)

    ch_names = np.array(ch_names)
    ch_pixels = np.array(ch_pixels)
    classes = np.array(classes)

    #print(ch_names)
    #print(classes)
    #print(type(ch_names))
    #print(result)
    return ch_names, ch_pixels, classes


def read_train_sets(train_path, fonts, validation_size):
    """
        Input:
            - path to files,
            - list of file names in .csv
            - size of validation (e.g., 0.1)

        Output:
            - Datasets object containing training and validation
                dataset.
    """
    class DataSets(object):
        """
        Class for the number of datasets.
        Later used to hold dataset for testing and validation
        """
        pass
    data_sets = DataSets()

    # load data using path and the list of file names
    ch_names, ch_pixels, classes = load_data(train_path, fonts)

    # shuffling input data randomly
    ch_names, ch_pixels, classes = shuffle(ch_names, ch_pixels, classes)

    # get absolute number of validation size from validation ratio
    if isinstance(validation_size, float):
        validation_size = int(validation_size * ch_names.shape[0])

    # get validation dataset using validation size
    validation_images = ch_pixels[:validation_size]
    validation_labels = ch_names[:validation_size]
    validation_classes = classes[:validation_size]

    # get training dataset from remaining dataset
    train_images = ch_pixels[validation_size:]
    train_labels = ch_names[validation_size:]
    train_classes = classes[validation_size:]

    # put two datasets in datasets object
    data_sets.train = DataSet(train_labels, train_images, train_classes)
    data_sets.valid = DataSet(validation_labels, validation_images, validation_classes)

    return data_sets


# Unit Test
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    train_path = 'Dataset'
    fonts = os.listdir(train_path)
    #print(fonts)
    ch_names, ch_pixels, classes = load_data(train_path, fonts)
    print(ch_names[0])
    plt.imshow(ch_pixels[0])
    plt.show()

