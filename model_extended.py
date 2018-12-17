import tensorflow as tf


class MLPClassifierModel():

    """Basic blueprint for an Multi-layer perceptron model; note that this is in no way inteded as a finished product, but rather it is simply designed to demonstrate how a basic MLP generally is constructed and how it works on a fundamental level. It is highly recommended to add some form of regularization (such as dropout or L2 /L1 regularization methods) and some form of training validation set mechanism. Additionally, larger data sets can benefit from some form of batch optimization routine. Since the model is aimed to be as general as possible, no accuracy operator/test was included; hence, some accuracy metric should also be added into the self.build() routine. Note that the choice of metric is very problem-specific and hence should be added on a project to project basis.

    Note that the TensorFlow low level-API is used to construct this model; the same could be accomplished far quicker and far easier with layers or Keras.

    Parameters
    ----------
    n_features: int
        number of features present in each sample
    n_classes: int
        number of classes in targets
    eta: float
        learning rate of model
    n_epochs: int
        number of desired epochs
    random_state: int
        seed used for random number generator
    hidden_layers: dict object
        used to dictate the structure of the hidden layers. n_layers refers to the number of hidden layers and unit_count must be iterable and gives the number of units within each hidden layer; note that the counts in unit_count must be passed down in the correct order. 'activation' refers to the activation function to be used for each layer. Again, these must appear in the desired order of application.
    """

    def __init__(self, n_features, n_classes, hidden_layers={'n_layers': 2, 'unit_count': (50, 25), 'activation': None}, eta=0.01, n_epochs=100, random_state=1):

        self.eta = eta
        self.n_epochs = n_epochs
        self.random_state = random_state

        # the graph is then constructed; note that the graph itself does not need to be saved, and neither do the components. As long as the session is saved (self.sess) and instantiated with the graph, the model will work.

        g = tf.Graph()

        with g.as_default():

            tf.set_random_seed(random_state)

            self.build(n_features, n_classes, hidden_layers)

            self.init_op = tf.global_variables_initializer()

            self.saver = tf.train.Saver()

        self.sess = tf.Session(graph=g)

    def fc_layer(self, input_tensor, name, n_units, activation_fn=None):

        """Method used to generate a fully connected layer. Note on the weight matrix; the matrix has dimensions [n_features, n_units] i.e. each unit has its own weight vector, which are combined to form a resultant weight matrix. Conceptually, the number of units in the hidden layer corresponds to the number of dimensions the sample will have in the new feature space. Fhe final outputed layer then has dimensions [n_samples, n_units]; hence the MLP is a mapping function (i.e. kernel). It takes some input matrix and projects it unto a higher dimensional subspace. Each neuron within the layer produces one column vector (i.e. one output scalar for each inputed sample combined into a column vector) and the column vectors from all units are combined to produce an output.

        Parameters
        ----------
        input_tensor: tensor-object
            input tensor
        name: str
            name used for scope
        n_units: int
            total number of neurons within the layer
        activation_fn: function object (default=None)
            activation function
        """

        with tf.variable_scope(name):

            input_shape = input_tensor.get_shape().as_list()

            weights_shape = (input_shape[1], n_units)

            weights = tf.get_variable(name='weights', shape=weights_shape)

            print(weights)

            print(weights)
            biases = tf.get_variable(
                name='biases', initializer=tf.zeros(shape=weights_shape[1]))

            layer = tf.matmul(input_tensor, weights, name='layer')
            layer = tf.nn.bias_add(layer, biases)

            if activation_fn is not None:
                return activation_fn(layer)
            else:
                return layer

    def build(self, n_features, n_classes, hidden_layers):

        """Method used to build the model

        Parameters
        ----------
        n_features: int
            number of features present in each sample
        n_classes: int
            number of classes
        """

        # a series of placeholders are first defined

        tf_x = tf.placeholder(tf.float32, shape=(
            None, n_features), name='tf_x')
        tf_y = tf.placeholder(tf.int32, shape=(None), name='tf_y')

        # the target variables are one hot encoded

        tf_y_onehot = tf.one_hot(
            indices=tf_y, depth=n_classes, name='tf_y_onehot')

        # a placeholder for the dropout rate and the regularizaton constant are also defined

        keep_proba = tf.placeholder(tf.float32, name='keep_proba')

        tf_lambda = tf.placeholder(tf.float32, name='lambda')

        # the layers are then constructed

        layers = {}

        for i, count in enumerate(hidden_layers['unit_count']):

            # the hidden layers are named as h1, h2, h3... etc

            name = 'h{}'.format(i+1)

            # the first layer takes the input layer as an input

            if name == 'h1':
                layers[name] = self.fc_layer(
                    input_tensor=tf_x, name=name, n_units=count, activation_fn=hidden_layers['activation'][i])
                layers[name] = tf.nn.dropout(layers[name], keep_proba)

            # all subsequent layers take the previously defined layers as the input tensor. the 'tag' variable simple refers to the name of the previous layer

            else:
                tag = 'h{}'.format(i)
                layers[name] = self.fc_layer(
                    input_tensor=layers[tag], name=name, n_units=count, activation_fn=hidden_layers['activation'][i])
                layers[name] = tf.nn.dropout(layers[name], keep_proba)

        # the ouput layer is then defined

        output = self.fc_layer(input_tensor=layers['h{}'.format(i+1)], name='output', n_units=n_classes, activation_fn=None)

        # a prediction operation is then defined; note that both probabilites or labels can be returned

        y_pred = {'probabilities': tf.nn.softmax(output, name='probabilities'), 'labels': tf.cast(
            tf.argmax(output, axis=1), tf.int32, name='labels')}

        # the cost function is defined

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=output, labels=tf_y_onehot), name='cost')

        # an l2 regularization term is then also added. Note that, in order to retrieve the weight variables from the other scopes, the reuse=True argument must be given.

        r = 0

        # the scope names are given by the layers dictionary keys

        for name in layers.keys():

            # the weight vectors are then retrieved form the various scopes

            with tf.variable_scope(name, reuse=True):

                weight = tf.get_variable(name='weights')

                r += tf.nn.l2_loss(weight)

        # finally, the cost function is adjusted by adding the regularization term multiplied by the lambda constant

        cost = tf.add(cost, r*tf_lambda)

        # finally a training operator is defined using the Adam Routine

        train_op = tf.train.AdamOptimizer(
            learning_rate=self.eta).minimize(cost, name='train_op')


    def train(self, X, y, l2_strength=0.):

        """Method used to train the model over the given set of epochs

        Parameters
        ----------
        X: array-like, shape=[n_samples, n_features]
            training data
        y: array-like, shape=[n_samples]
            target values
        l2_strength: float
            l2 regulzariation constant

        """

        self.sess.run(self.init_op)

        self.training_cost = []

        for epoch in range(1, self.n_epochs + 1):

            feed = {'tf_x:0': X, 'tf_y:0': y, 'keep_proba:0': 0.5, 'lambda:0': l2_strength}

            c, _ = self.sess.run(['cost:0', 'train_op'], feed_dict=feed)

            self.training_cost.append(c)

            print('Epoch: {} Avg. Cost: {}'.format(epoch, c))

    def predict(self, X, proba=False):

        """Method to return the models prediction from some data set

        Parameters
        ----------
        X: array-like
            data set
        proba: boolean
            return probabilities if set to True, returns labels otherwise
         """

        feed = {'tf_x:0': X, 'keep_proba:0': 1}

        if proba:
            return self.sess.run('probabilities:0', feed_dict=feed)
        else:
            return self.sess.run('labels:0', feed_dict=feed)
