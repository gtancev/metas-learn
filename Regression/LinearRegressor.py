__author__ = "Georgi Tancev"
__copyright__ = "© Georgi Tancev"


import numpy as np


class LinearRegressor:
    """
    Linear regression implementation.
    It uses stochastic gradient descent (SGD) with random sampling of batches.
    Inputs: batch_size: size of batch in SGD steps
            alpha: L2-regularization parameter in SGD steps
            learning_rate: learning rate in SGD steps
            n_iterations: number of SGD iterations
            warm_start: whether to use old parameters as starting point
    """
    def __init__(self, batch_size=300, alpha=0.0, learning_rate=1e-3, n_iterations=1e4, warm_start=False):
        self.batch_size = batch_size
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = int(n_iterations)
        self.warm_start = warm_start

    @staticmethod
    def random_normal(shape, loc=0.0, scale=1.0):
        """
        Generates random normal numbers of specified shape.
        Input:  shape: shape as tuple
                loc: mean of distribution
                scale: standard deviation
        Output: normally distributed array
        """
        return np.random.normal(loc, scale, shape)

    @staticmethod
    def mean_squared_error(y_pred, y_true):
        """
        Compute mean squared error loss.
        Input:  y_pred: prediction
                y_true: true values
        Output: loss
        """
        e = np.subtract(y_pred, y_true)
        return np.transpose(np.sum(np.multiply(e, e), axis=1))

    @staticmethod
    def sample(low, high, size):
        """
        Samples a batch from the data set.
        Input:  low: lowest number
                high: highest number
                size: number of instances
        Output: index of samples
        """
        return np.random.randint(low, high, size)

    def _initialize_coeffs(self, n_features, n_outputs):
        """
        Initialize coefficients.
        Input:  n_features: number of features in X
                n_outputs: number of outputs in Y
        Output: none
        """
        self.coeffs = self.random_normal((n_outputs, n_features))
        self.intercept = self.random_normal((n_outputs, 1))
        return

    def _forward(self, X):
        """
        Compute prediction of samples X.
        Input:  X: matrix with samples
        Output: y: prediction of samples
        """
        return np.add(np.dot(self.coeffs, X), self.intercept)

    def _compute_gradients(self, X, y):
        """
        Method to compute gradients.
        Input:  X: matrix with samples
                y: matrix with outputs
        Output: gradients
        """
        beta = - 2 / self.batch_size
        dW = beta * np.dot(np.subtract(y, self._forward(X)), np.transpose(X))
        db = beta * np.sum(np.subtract(y, self._forward(X)))
        return (dW, db)

    def _gradient_descent(self, gradients):
        """
        Gradient descent update rule.
        Input:  gradients
        Output: none
        """
        beta = (1 - self.alpha * self.learning_rate / self.batch_size)

        dW, db = gradients

        self.coeffs = np.subtract(beta * self.coeffs, self.learning_rate * dW)
        self.intercept = np.subtract(beta * self.intercept, self.learning_rate * db)
        return

    def fit(self, X, y):
        """
        Method to fit the neural network regressor.
        Input:  X: data matrix of shape (n_samples, n_features).
                y: labels of shape (n_samples)
        Output: none
        """
        # convert to array if not yet
        X = np.transpose(np.asarray(X))
        y = np.transpose(np.array(y, ndmin=2))

        n_points = X.shape[1]  # number of instances in X
        n_points_ = y.shape[1]  # number of instances in y

        try:
            assert n_points == n_points_
        except AssertionError:
            y = np.transpose(y)

        n_features = X.shape[0]  # number of features in X
        n_outputs = y.shape[0]  # number of outputs in y
        
        # initialize array for loss
        self.loss = np.empty((int(self.n_iterations), int(n_outputs)), dtype=float)

        if (not self.warm_start):
            # initialize coefficients
            self._initialize_coeffs(n_features, n_outputs)

        if (self.batch_size > n_points):
            # adjust batch size if necessary
            self.batch_size = n_points
            print("""The batch size is larger than the
                number of samples and is adjusted accordingly.""")

        for i in range(self.n_iterations):

            # generate a batch
            sample = self.sample(0, n_points, self.batch_size)

            # predict
            y_pred = self._forward(X[:, sample])

            # compute cost
            cost = self.mean_squared_error(y_pred, y[:, sample])

            # add cost to list
            self.loss[i, :] = cost

            # compute gradients
            gradients = self._compute_gradients(X[:, sample], y[:, sample])

            # update parameters
            self._gradient_descent(gradients)
        return

    def predict(self, x):
        """
        Predict the output of sample x.
        Input:  x: new sample
        Output: y: label
        """
        try:
            y_pred = self._forward(np.transpose(np.asarray(x)))
            return np.transpose(y_pred)
        except AttributeError:
            print("Train model first.")
