"""
Class to read the input data.
reads cvs file from provided training and testing data folders.
"""

import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

class DataSet(object):
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
                # r = (temp >> 5) * 32
                # g = ((temp & 28) >> 2) * 32
                # b = (temp & 3) * 64

                r = temp
                g = temp
                b = temp
                result[:, :, 0] = r
                result[:, :, 1] = g
                result[:, :, 2] = b

                ch_pixels.append(result)

            # ch_names

    unique_classes = np.unique(classes)
    for character_int in classes:
        temp = np.zeros(unique_classes.size)
        #index = np.where(classes == character_int)
        #print(character_int)
        index = character_int - 65
        temp[index] = 1.0
        #print(temp)
        ch_names.append(temp)

    ch_names = np.array(ch_names)
    ch_pixels = np.array(ch_pixels)
    classes = np.array(classes)
        #ch_names.append(names)
        #ch_pixels.append(result)
        #font_labels.append(font[:-4])

        #ch_names = np.array(ch_names)
        #ch_pixels = np.array(ch_pixels)
        #font_labels = np.array(font_labels)
        #print(ch_pixels)

    #print(ch_names)
    #print(classes)


    #print(type(ch_names))
    #print(result)
    return ch_names, ch_pixels, classes

def read_train_sets(train_path, fonts, validation_size):
    class DataSets(object):
        pass
    data_sets = DataSets()

    ch_names, ch_pixels, classes = load_data(train_path, fonts)
    ch_names, ch_pixels, classes = shuffle(ch_names, ch_pixels, classes)
    if isinstance(validation_size, float):
        validation_size = int(validation_size * ch_names.shape[0])

    validation_images = ch_pixels[:validation_size]
    validation_labels = ch_names[:validation_size]
    validation_classes = classes[:validation_size]

    train_images = ch_pixels[validation_size:]
    train_labels = ch_names[validation_size:]
    train_classes = classes[validation_size:]

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

