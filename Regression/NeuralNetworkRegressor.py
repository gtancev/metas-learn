__author__ = "Georgi Tancev"
__copyright__ = "© Georgi Tancev"


import numpy as np


class NeuralNetworkRegressor:
    """
    Neural network regression implementation (error backpropagation).
    It uses stochastic gradient descent (SGD) with random sampling of batches.
    Inputs: hidden_layer_sizes: tuple with sizes of each hidden layer
            batch_size: size of batch in SGD steps
            alpha: L2-regularization parameter in SGD steps
            learning_rate: learning rate in SGD steps
            n_iterations: number of SGD iterations
            warm_start: whether to use old parameters as starting point
    """
    def __init__(self, hidden_layer_sizes=(10,), batch_size=300, alpha=0.0, learning_rate=1e-3, n_iterations=1e4, warm_start=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = int(n_iterations)
        self.warm_start = warm_start
        self.coeffs = {}

    @property
    def num_hidden_layers(self):
        """
        Print number of hidden layers.
        """
        return len(self.hidden_layer_sizes)

    @staticmethod
    def sigmoid(Z):
        """
        Sigmoid activation function.
        Input:  Z: weightes latent variables
        Output: A: activations
        """
        return 1.0 / (1.0 + np.exp(-Z))

    @staticmethod
    def linear(Z):
        """
        Linear activation function.
        Input:  Z: weightes latent variables
        Output: A: activations
        """
        return Z

    @staticmethod
    def relu(Z):
        """
        Rectifier activation function.
        Input:  Z: weighted latent variables
        Output: A: activations
        """
        return np.maximum(0, Z)

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
        Initialize coefficients for all layers.
        Input:  n_features: number of features in X
                n_outputs: number of outputs in X
        Output: none
        """
        n_layers = self.num_hidden_layers

        # input layer
        n_latent = self.hidden_layer_sizes[0]
        W = self.random_normal((n_latent, n_features))
        b = self.random_normal((1, 1))
        self.coeffs["W0"] = W
        self.coeffs["b0"] = b

        # middle layers
        for layer in range(1, n_layers):
            n_latent = self.hidden_layer_sizes[layer]
            n_latent_last = self.hidden_layer_sizes[layer - 1]
            W = self.random_normal((n_latent_last, n_latent))
            b = self.random_normal((1, 1))
            self.coeffs["W" + str(layer)] = W
            self.coeffs["b" + str(layer)] = b

        # output layer
        n_latent = self.hidden_layer_sizes[-1]
        W = self.random_normal((n_outputs, n_latent))
        b = self.random_normal((1, 1))
        self.coeffs["W" + str(n_layers)] = W
        self.coeffs["b" + str(n_layers)] = b
        return

    def _forward(self, A, W, b, activation_fn):
        """
        Passes information forward.
        Inputs: A: activations from previous layer
                W, b: parameters
                activation_fn: activation function to use
        Output: Z: linear combinations
                cache: tuple of stored values
        """
        Z = np.add(np.dot(W, A), b)
        A_next = activation_fn(Z)
        cache = ((A, W, b), Z)
        return A_next, cache

    def _forward_pass(self, X):
        """
        Forward pass of signal / data.
        Input:  X: data
        Output: Y: prediction
                caches: stored values
        """
        A = X
        caches = []
        n_layers = self.num_hidden_layers

        # first n_layers - 1 use ReLU activation
        for layer in range(n_layers):
            A_prev = A
            A, cache = self._forward(
                A_prev, self.coeffs["W" + str(layer)], self.coeffs["b" + str(layer)],
                self.relu)
            caches.append(cache)

        # last layer uses linear activation
        y_pred, cache = self._forward(
            A, self.coeffs["W" + str(n_layers)], self.coeffs["b" + str(n_layers)],
            self.linear)
        caches.append(cache)
        return y_pred, caches

    def _relu_gradient(self, dA, Z):
        """
        Gradient of rectifier activation.
        Input:  dA: derivatives
                Z: latent variables
        Output: dZ: delta of weighted previous activations
        """
        A = self.relu(Z)
        dZ = np.multiply(dA, np.int64(A > 0))
        return dZ

    def _linear_gradient(self, dA, Z):
        """
        Gradient of linear activation.
        Input:  dA: derivatives
                Z: latent variables
        Output: dZ: delta of weighted previous activations
        """
        dZ = dA
        return dZ

    def _backward(self, dA, cache, activation_fn):
        """
        Passes information backwards.
        Inputs: dA: derivatives
        Output: cache: stored values
        """
        (A_last, W, _), Z = cache
        dZ = activation_fn(dA, Z)
        beta = (1 / self.batch_size)

        dW = beta * np.dot(dZ, np.transpose(A_last))
        db = beta * np.sum(dZ, axis=1, keepdims=True)
        dA_last = np.dot(np.transpose(W), dZ)
        return dA_last, dW, db

    def _backward_pass(self, y_pred, y_true, caches):
        """
        Passes loss backwards, i.e. error backpropagation.
        Input:  y_pred: prediction
                y_true: ground truth
        Output: none
        """
        n_layers = self.num_hidden_layers + 1
        gradients = {}

        dy_pred = np.subtract(y_pred, y_true)

        gradients["dA" + str(n_layers - 1)], gradients["dW" + str(n_layers - 1)], gradients[
        "db" + str(n_layers - 1)] = self._backward(
            dy_pred, caches[n_layers - 1], self._linear_gradient)

        for layer in range(n_layers - 1, 0, -1):
            current_cache = caches[layer - 1]
            gradients["dA" + str(layer - 1)], gradients["dW" + str(layer - 1)], gradients[
                "db" + str(layer - 1)] = self._backward(
                    gradients["dA" + str(layer)], current_cache,
                    self._relu_gradient)
        return gradients

    def _gradient_descent(self, gradients):
        """
        Gradient descent update rule.
        Input:  gradients
        Output: none
        """
        n_layers = self.num_hidden_layers
        beta = (1 - self.alpha * self.learning_rate / self.batch_size)

        for layer in range(0, n_layers + 1):
            self.coeffs["W" + str(layer)] = np.subtract(beta * self.coeffs[
                "W" + str(layer)], self.learning_rate * gradients["dW" + str(layer)])
            self.coeffs["b" + str(layer)] = np.subtract(beta * self.coeffs[
                "b" + str(layer)], self.learning_rate * gradients["db" + str(layer)])
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

        if (not self.warm_start) or (len(self.coeffs) == 0):
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

            # iterate over layers to get the final output and the cache
            y_pred, caches = self._forward_pass(X[:, sample])

            # compute cost
            cost = self.mean_squared_error(y_pred, y[:, sample])

            # add cost to list
            self.loss[i, :] = cost

            # iterate over layers backward to get gradients
            gradients = self._backward_pass(y_pred, y[:, sample], caches)

            # update parameters
            self._gradient_descent(gradients)
        return

    def predict(self, x):
        """
        Predict the output of sample x.
        Input:  x: new sample
        Output: y: label
        """
        if len(self.coeffs):
            y_pred, _ = self._forward_pass(np.transpose(np.asarray(x)))
            return np.transpose(y_pred)
        else:
            print("Train model first.")
