__author__ = "Georgi Tancev"
__copyright__ = "© Georgi Tancev"


import numpy as np


class StandardScaler:
    """
    Class to scale data by subtracting mean and
    dividing by standard deviation (Z-scores).
    """
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X):
        """
        Method to find means and standard deviations.
        Input:  X: data
        Output: none
        """
        # mean
        if self.with_mean:
            self.mean = np.mean(X, axis=0, keepdims=True)
        else:
            self.mean = 0
        # standard deviation
        if self.with_std:
            self.std = np.std(X, axis=0, keepdims=True)
        else:
            self.std = 1
        return

    def transform(self, X):
        """
        Method to transform data.
        Input:  X: data
        Output: Z: z-scored data
        """
        return np.divide(np.subtract(X, self.mean), self.std)

    def fit_transform(self, X):
        """
        Method to find means and standard deviations
        and directly transform data.
        Input:  X: data
        Output: Z: z-scored data
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z):
        """
        Method to inverse-transform data.
        Input:  Z: z-scored data
        Output: X: data
        """
        return np.add(np.multiply(Z, self.std), self.mean)
