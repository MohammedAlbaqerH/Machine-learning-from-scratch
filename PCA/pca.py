import numpy as np


class PCA:

    def __init__(self, dim = 2):
        self.dim = dim
        self.components = None
        self.mean = None

    def fit(self, X):
        #callate mean
        self.mean = X.mean(axis = 0)
        self.std = X.std(axis = 0)
        #callate coverance matrix
        X = (X - self.mean)/(self.std)
        cov = X.T@X
        #calculate eigvactor, eigvalue
        eigvalue, eigvactor = np.linalg.eig(cov)
        eigvactor = eigvactor.T
        idxs = np.argsort(eigvalue)[::-1]
        eigvalue  = eigvalue[idxs]
        eigvactor = eigvactor[idxs]
        self.components = eigvactor[0:self.dim]


    def transform(self, X):
        X = X - self.mean
        X = np.dot(X, self.components.T)
        return X

