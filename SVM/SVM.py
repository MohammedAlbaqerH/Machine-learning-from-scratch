import numpy as np
import pickle



class SVM:
    
    def __init__(self, lam = 0.0001, lr = 0.0001, epochs = 100):
        self.lr = lr
        self.lam = lam
        self.epochs = epochs

        self.learnd = False

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):

            for idx, x_i in enumerate(X):
                yh = 1 - y[idx]*(np.dot(self.w, x_i) - self.b)
                if yh <= 0:
                    self.w -= self.lr*(2*self.lam*self.w)
                else:
                    self.w -= self.lr*((-y[idx]*x_i) + self.lr*(2*self.lam*self.w))
                    self.b -= self.lr*(y[idx])


    def _predict(self, x):
        y = np.dot(x, self.w) - self.b
        return np.where(y <= 0, -1, 1)
                


def save(d, model):
    par = {"w":model.w,  "b":model.b}
    file = open(d, 'wb')
    pickle.dump(par, file)
    file.close()


def load(d):
    with open(d, 'rb') as f:
        par = pickle.load(f)
    try:
        svm = SVM()
        svm.w = par["w"]
        svm.b = par["b"]
    except:
        print("you must have SVM class")
    return svm

                

    