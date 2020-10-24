import sklearn.datasets as datasets
import numpy as np
from  SVM import SVM, save, load
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from DecisionTree import *

data = datasets.load_breast_cancer()
X  = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
# svm = SVM()
# svm.fit(x_train, y_train)
tree = Tree(max_depth=100)

tree.fit(x_train, y_train)

print(tree.predict(x_test) - y_test)

# save("svm", svm)