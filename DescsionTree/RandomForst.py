import numpy as np
from collections import Counter
from DecisionTree import Tree

def bootstrab(X, y):
    n_sample = X.shape[0]
    idexs = np.random.choice(n_sample,n_sample, replace=True)
    return X[idexs], y[idexs]

def most_commen_val(y):
    c = Counter(y)
    most = c.most_common(1)
    return most[0][0]

class RandomForst:

    def __init__(self, num_tree = 100, min_split_sample = 2, max_depth = 100, n_feat = None):
        self.num_tree = num_tree
        self.min_split_sample = min_split_sample
        self.max_depth = max_depth
        self.n_feat = n_feat

    def fit(self, X, y):
        self.tree = []
        for _ in range(self.num_tree):
            tree = Tree(self.min_split_sample, self.max_depth, self.n_feat)
            x_sample, y_sample  = bootstrab(X, y)
            tree.fit(x_sample, y_sample)
            self.tree.append(tree)
        
    def predict(self, X):
        tree_predict = np.array([tree.predict(X) for tree in self.tree])
        tree_predict = np.swapaxes(tree_predict, 0, 1)
        y_pred = [most_commen_val(y) for y in tree_predict]
        return y_pred


        


