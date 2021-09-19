from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
iris = datasets.load_iris()
X = iris.data
y = iris.target
# print(X,y)

# print(X.shape)
# print(y.shape)
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)