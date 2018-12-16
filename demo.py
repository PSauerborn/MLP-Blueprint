from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from model import MLPClassifierModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# the following code is desinged as a demo; the iris data set is used as a simple example

iris = datasets.load_iris()

X, y = iris.data, iris.target

# the data is first split and standardized

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1, stratify=y)

s = StandardScaler()

X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)

# the number of features and classes are evaluated

n_features, n_classes = X.shape[1], np.unique(y).shape[0]

# the structure of the MLP is then defined with the following dictionary. Note that it consists of 2 hidden layers, the first having 50 units and the second having 25. Both use the relu function as an activation function

hidden_layers = {'n_layers': 2, 'unit_count': (50, 25), 'activation': (tf.nn.relu, tf.nn.relu)}

# the model is then instaniated

model = MLPClassifierModel(n_features=n_features, n_classes=n_classes, hidden_layers=hidden_layers, eta=0.02, n_epochs=200)

# and the model is trained. A set of predictions is then also made for testing

model.train(X_train, y_train)
pred = model.predict(X_test)

# the average training cost is then plotted for each epoch

fig, ax = plt.subplots()
ax.plot(model.training_cost)
ax.set_xlabel('Epoch')
ax.set_ylabel('Average Training Cost')
plt.show()

# note that model achieves an accuracy of 80% without any modification. This could be greatly improved by altering the basic model slightly (by adding regularization methods for example)

print((pred[pred == y_test].shape[0] / y_test.shape[0])*100 )
