# MLP-Blueprint
The code contains a stripped down blueprint for an MLP Classifier; note that it is not designed to be a finished product. It simply gives a starting point for the construction of a more complex model

The architecture is defined with a dictionary which takes the form {'n_layers': n, 'unit_count': m, 'activation': func}. 
n_layers defines the number of hidden layers the model generates, and unit_count gives the number of units in each layer. Note that these must be given in the form of some iterator (tuple, list, etc) and must appear in the desired order

activation is used to define the type of activation function (if any) to be used. Again, these must appear in the correct order, if specified. Additionally, if one activation function is defined, the activation function for all layers must be defined i.e. one cant define 3 layers and then only pass one activation function.

For example, to construct an MLP with 3 layers, the first having 50 units, the second having 100, and the third having 25, all with the relu activation function, one simply passes down

hidden_layers = {'n_layers': 3, 'unit_count': (50, 100, 25), 'activation': (tf.nn.relu, tf.nn.relu, tf.nn.relu)}

A demo version is also included which is trained using the iris data set. Out of the box, it achieves an accuracy of 80%; this can be greatly improved by making a few simple adjustments (such as adding some regularization routine)

Again, this is not intended as a finished product; it is simply designed as a starting point, particulary for people new to TensorFlow (such as myself)

Update 17/12/2018
-----------------

SLight extension was made to the model to include L2 regularization and dropout to control overfitting
