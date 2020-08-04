__author__ = "Georgi Tancev"
__copyright__ = "© Georgi Tancev"


import numpy as np


class GaussianMixtureModel:
    def __init__(self, n_components=2, rtol=1e-6, max_iter=50, restarts=10):
        """
        Creates GaussianMixtureModel object.
        Input:
        n_components: int, number of Gaussians (classes) in the mixture
        rtol: float, relative change for convergence check
        max_iter: int, maximum number of E- & M-step iterations
        restarts: int, number of different starting positions

        Output:
        Instance of GaussianMixtureModel.
        """
        self.n_components = n_components
        self.rtol = rtol
        self.max_iter = max_iter
        self.restarts = restarts
        self.best_loss = None
        self.best_pi = None
        self.best_mu = None
        self.best_sigma = None
        self.best_gamma = None

    def _E_step(self, X, pi, mu, sigma):
        """
        Performs E-step on GMM model.
        Input:
        X: (N x d), data points
        pi: (C), mixture component weights
        mu: (C x d), mixture component means
        sigma: (C x d x d), mixture component covariance matrices

        Output:
        gamma: (N x C), probabilities of clusters for objects
        """

        N = X.shape[0]  # number of objects
        C = pi.shape[0]  # number of clusters
        d = mu.shape[1]  # dimension of each object
        gamma = np.zeros((N, C))  # posterior distribution, responsibilities

        for n in range(N):
            x_i = np.expand_dims(X[n, :], axis=0)
            for c in range(C):
                mu_c = np.expand_dims(mu[c, :], axis=0)
                B = np.linalg.inv(sigma[c, :, :])  # precision matrix
                D = np.linalg.det(sigma[c, :, :])  # determinant
                z_i = -1/2 * np.matmul((mu_c - x_i), np.matmul(B, np.transpose(mu_c - x_i)))
                gamma[n, c] = (pi[c] * (1 / np.sqrt(((2 * np.pi)**d) * D))) * np.exp(z_i)

        Z = np.expand_dims(np.sum(gamma, axis=1), axis=-1)
        gamma = gamma / Z
        return gamma

    def _M_step(self, X, gamma):
        """
        Performs M-step on GMM model
        Input:
        X: (N x d), data points
        gamma: (N x C), distribution q(T)

        Output:
        pi: (C)
        mu: (C x d)
        sigma: (C x d x d)
        """

        N = X.shape[0]  # number of objects
        C = gamma.shape[1]  # number of clusters
        d = X.shape[1]  # dimension of each object

        sigma = np.zeros((C, d, d))
        N_k = np.expand_dims(np.sum(gamma, axis=0), axis=-1)

        pi = N_k / N
        mu = (1 / N_k) * np.matmul(np.transpose(gamma), X)
        for c in range(C):
            E = np.zeros((d, d))
            for n in range(N):
                z = np.expand_dims(X[n, :] - mu[c, :], -1)
                E = E + gamma[n, c] * np.matmul(z, np.transpose(z))
            sigma[c, :, :] = (1 / N_k[c]) * E
        return pi, mu, sigma

    def _compute_vlb(self, X, pi, mu, sigma, gamma):
        """
        Compute the value of the variational lower bound.
        Input:
        X: (N x d), data points
        gamma: (N x C), distribution q(T)
        pi: (C)
        mu: (C x d)
        sigma: (C x d x d)

        Output:
        Value of variational lower bound (VLB).
        """

        N = X.shape[0]  # number of objects
        C = gamma.shape[1]  # number of clusters
        d = X.shape[1]  # dimension of each object

        loss = 0  # initialize loss
        eps = 1e-12  # zero division inhibitor

        for n in range(N):
            x_i = X[n, :]
            for c in range(C):
                mu_c = mu[c, :]
                B = np.linalg.inv(sigma[c, :, :])  # precision matrix
                D = np.linalg.det(sigma[c, :, :])  # determinant
                z_i = -1/2 * np.matmul((mu_c - x_i), np.matmul(B, np.transpose(mu_c - x_i)))  # z-score
                p_i = (1 / np.sqrt(((2 * np.pi)**d) * D)) * np.exp(z_i)  # responsibilities
                loss = loss + gamma[n, c] * (np.log(pi[c] + eps) + np.log(p_i + eps)) - gamma[n, c] * np.log(gamma[n, c] + eps)
        return loss

    def fit(self, X):
        '''
        Starts with random initialization *restarts* times
        Runs optimization until saturation with *rtol* reached
        or *max_iter* iterations were made.
        Input:
        X: (N, d), data points
        '''

        # Define hyperparameters.
        C = self.n_components
        rtol = self.rtol
        max_iter = self.max_iter
        restarts = self.restarts

        N = X.shape[0]  # number of objects
        d = X.shape[1]  # dimension of each object
        best_loss = np.inf  # initialize best loss
        best_pi = None  # best responsibilities
        best_mu = None  # best means
        best_sigma = None  # best covariances

        for _ in range(restarts):
            try:

                # Initialize.
                pi = np.zeros((C))
                pi[0 : (C-1)] = np.random.uniform(0.1, 1 / C, (C - 1))
                pi[-1] = 1 - np.sum(pi)

                mu = np.mean(X, axis=0) + np.random.randn(C, d)

                I_matrix = np.eye(N=d, M=d)  # eye matrix
                sigma_0 = np.std(X)  # guess for sigma
                E = np.expand_dims(sigma_0**2 * I_matrix, axis=0)
                sigma = np.repeat(E, C, axis=0)

                new_loss = np.inf  # loss

                for _ in range(max_iter):

                    # Perform E- and M-step.
                    gamma = self._E_step(X, pi, mu, sigma)
                    pi, mu, sigma = self._M_step(X, gamma)

                    # Compute loss.
                    current_loss = new_loss
                    new_loss = self._compute_vlb(X, pi, mu, sigma, gamma)

                    # Check for convergence.
                    if current_loss < np.inf:
                        rcurrent = (current_loss - new_loss) / (current_loss)
                        if np.abs(rcurrent) <= rtol:
                            if new_loss < best_loss:
                                best_loss = new_loss
                                best_pi = pi
                                best_mu = mu
                                best_sigma = sigma
                            break

            except np.linalg.LinAlgError:
                print("Singular matrix: components collapsed")
                pass

        # Save result.
        self.best_loss = best_loss
        self.best_pi = best_pi
        self.best_mu = best_mu
        self.best_sigma = best_sigma
        self.best_gamma = self._E_step(X, best_pi, best_mu, best_sigma)
        return

    def predict(self, x):
        """
        Predict responsibilities of sample x.
        Input:
        x: (1, d)

        Output:
        responsibilities
        """
        x = np.asarray(x)
        try:
            return self._E_step(x, self.best_pi, self.best_mu, self.best_sigma)
        except AttributeError:
            print("Train model first.")
