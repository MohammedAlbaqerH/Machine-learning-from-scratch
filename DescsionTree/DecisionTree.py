import numpy as np
from collections import Counter

def entropy(d):
    c = np.bincount(d)
    c = c[c != 0]
    prop = c/len(d)
    E = - prop*np.log2(prop)
    return np.sum(E)

class Node:
    def __init__(self, beast_split_feature = None, threashold = None, left = None,right = None,*,value = None):
        self.beast_split_feature = beast_split_feature
        self.threashold  = threashold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf(self):
        return self.value is not None



class Tree:
    def __init__(self, min_split_sample = 2, max_depth = 100, n_feat = None):
        self.min_split_sample = min_split_sample
        self.max_depth = max_depth
        self.n_feat = n_feat
        self.root = None

    def fit(self, X,y):
        _, n_features = X.shape
        self.n_feat = n_features if  not self.n_feat else min(n_features, self.n_feat)
        self.root = self._grow_tree(X,y)


    def predict(self, X):
        return np.array([self._trav(x, self.root) for x in X])
    
    def _trav(self, x, node):
        if node.is_leaf():
            return node.value
        elif x[node.beast_split_feature] <= node.threashold:
            return self._trav(x, node.left)
        return self._trav(x, node.right)


    def _grow_tree(self, X, y, depth = 0):
        _, n_features = X.shape
        n_classes = len(np.unique(y))

        #stopping cratiria
        if (depth >= self.max_depth or n_classes <= self.min_split_sample
        or n_classes == 1):
            leaf_val = self._most_commen_val(y)
            return Node(value=leaf_val)

        feat_idx = np.random.choice(n_features, self.n_feat, replace=False)

        best_threash, best_feat = self._best_criteria(X, y, feat_idx)
        left_idx, right_idx = self._split(X[:, best_feat] ,best_threash)

        righTree = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)
        lefTree = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        return Node(best_feat, best_threash, lefTree, righTree)



    
    def _best_criteria(self, X, y, feat_idxs):
        best_gian = -1
        split_idx, split_thrishold = None, None

        for feat_idx in feat_idxs:
            X_cloumnd = X[:, feat_idx]
            thrisholds = np.unique(X_cloumnd)
            for thrishold in thrisholds:
                gain = self._info_gain(y, X_cloumnd, thrishold)

                if gain > best_gian:
                    best_gian = gain
                    split_idx = feat_idx
                    split_thrishold = thrishold
        return split_thrishold, split_idx
    
    def _info_gain(self, y, X_cloumnd, thrishold):
        #perant entropy
        perant_entropy = entropy(y)
        #split
        left_idx, right_idx = self._split(X_cloumnd, thrishold)
        if len(left_idx) ==0  or len(right_idx) ==0:
            return 0
        left_label = y[left_idx]
        right_label =  y[right_idx]
        l_n, r_n = len(left_label), len(right_label)
        child_entropy = (l_n/len(y))*entropy(left_label) + (r_n/len(y))*entropy(right_label)

        #wighted avg
        return perant_entropy - child_entropy
    
    def _split(self, X_cloumnd, thrishold):
        left = np.argwhere(X_cloumnd < thrishold).flatten()
        right = np.argwhere(X_cloumnd >= thrishold).flatten()
        return left, right
    
    def _most_commen_val(self, y):
        c = Counter(y)
        most = c.most_common(1)
        return most[0][0]
            


        