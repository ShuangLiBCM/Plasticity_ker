"""
    Define the dataset class
"""
import numpy as np


class Dataset(object):

    def __init__(self, x, y):
        """
        :param x: shape, sample_size * dimension
        :param y: shape, sample_size * 1
        """
        self.x = x
        self.y = y
        self._epochs_completed = 0
        self.sample_size = x.shape[0]
        self._index_in_epoch = 0

    def next_batch(self, batch_size, shuffle=True):
        """
        Return the next 'batch_size' examples from this dataset.
        :param batch_size:
        :param shuffle:
        :return:
        """
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self.sample_size)
            np.random.shuffle(perm0)
            self._x = self.x[perm0]
            self._y = self.y[perm0]

        # Go to the next epoch
        if start + batch_size > self.sample_size:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self.sample_size - start
            x_rest_part = self._x[start:self.sample_size]
            y_rest_part = self._y[start:self.sample_size]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self.sample_size)
                np.random.shuffle(perm)
                self._x = self.x[perm]
                self._y = self.y[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            x_new_part = self._x[start:end]
            y_new_part = self._y[start:end]
            return np.concatenate((x_new_part, x_rest_part), axis=0), np.concatenate((y_new_part, y_rest_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._x[start:end], self._y[start:end]

    def batch(self, batch_size):
        """
        Return a random batch of 'batch_size' examples from this dataset.
        :param batch_size:
        :return:
        """

        # Shuffle the index
        perm0 = np.arange(self.sample_size)
        np.random.shuffle(perm0)
        self._x = self.x[perm0]
        self._y = self.y[perm0]
        return self._x[:batch_size], self._y[:batch_size]