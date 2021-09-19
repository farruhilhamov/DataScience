from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

iris = datasets.load_iris()
classes = ["Iris Setosa","Iris Versicolour","Iris Verginica "]
X = iris.data
y = iris.target
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = svm.SVC()
model.fit(x_train,y_train)

print(model)
predictions = model.predict(x_test)
acc = accuracy_score(y_test,predictions)

print("predictions:",predictions)
print('actual:',y_test)
print("accuracy:",acc)

for i in range(len(predictions)):
    print(classes[predictions[i]])