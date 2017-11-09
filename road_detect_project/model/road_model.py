"""
version 0.5
2017.10.20
"""

import numpy as np
import tensorflow as tf

from abc import ABCMeta, abstractclassmethod

from road_detect_project.model.params import dataset_params, optimization_params
from road_detect_project.model.data import AerialDataset


class BaseLayer(object):
    """
    All network layers inherit from this class.
    Contains methods for initializing suitable random weights 
    and conducting dropout.
    """

    def __init__(self, input):
        self.input = input
        # self.rng = rng
        # self.dropout_rate = dropout_rate

    def set_bias(self, b, n, name):
        b_name = "b" if name is None else name + "_bias"
        if b is None:
            self.b = tf.Variable(tf.zeros([n]), name=b_name)
        else:
            self.b = b

    def set_weights_std(self, W, mean, std, size, name):
        w_name = "W" if name is None else name + "_weights"
        if W is None:
            initial = tf.truncated_normal(shape=size, mean=mean, stddev=std)
            self.W = tf.Variable(initial, name=w_name)
        else:
            self.W = W

    def set_weights_uniform(self, W, low, high, size, name):
        w_name = "W" if name is None else name + "_weights"
        if W is None:
            initial = tf.random_uniform(shape=size, minval=low, maxval=high)
            self.W = tf.Variable(initial, name=w_name)
        else:
            self.W = W

    def dropout(self):
        pass


class ConvPoolLayer(BaseLayer):
    """
    This class initialize a convolutional layer. 
    Parameters supplied in the init decide the number of kernels, 
    the kernel sizing, the activation function. 
    The layer can also initialize from existing weights
    and biases (Stored models and so forth). 
    The layer support dropout, strides, and max pooling.
    The pooling step is not treated as a separate layer, 
    but belongs to a convolutional layer. To deactivate pooling
    the poolsize should be set (1,1).
    """

    def __init__(self, input, filter_shape, activation=tf.nn.relu,
                 image_shape=None, padding="SAME", strides=(1, 1, 1, 1),
                 pool_size=(1, 2, 2, 1), pool_strides=(1, 2, 2, 1),
                 W=None, b=None, name=None, dropout=False, keep_prob=1.0, verbose=True):
        """
        :param input: image tensor
        :param filter_shape: (filter height, filter width,num input feature maps, number of filters)
        :param activation: Choice of activation function
        :param image_shape: (batch size, image height, image width,num input feature maps)
        :param padding: Zero padding
        :param strides: Step
        :param pool_size: The downsampling (pooling) factor (#rows, #cols)
        :param pool_strides: Zero padding
        :param W: Supplied layer weights.
        :param b: Supplied biases.
        :param verbose: Print layer arch. in console.
        """
        super(ConvPoolLayer, self).__init__(input)
        self._verbose(verbose, filter_shape, pool_size)

        if image_shape is not None:
            assert filter_shape[2] == image_shape[-1]
        # 一种比较简单、有效的方法是：权重参数初始化从区间均匀随机取值。

        fan_in = np.prod(filter_shape[:-1])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[-1] * np.prod(filter_shape[:2]) / np.prod(pool_size))

        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))

        self.set_weights_uniform(W, -W_bound, W_bound, filter_shape, name=name)
        self.set_bias(b, filter_shape[-1], name=name)

        conv_out = tf.nn.conv2d(input=self.input, filter=self.W,
                                strides=strides, padding=padding)
        out = activation(conv_out + self.b)

        pool_out = tf.nn.max_pool(value=out, ksize=pool_size,
                                  strides=pool_strides, padding=padding)
        if dropout:
            self.output = tf.nn.dropout(pool_out, keep_prob=keep_prob)
        else:
            self.output = pool_out

        self.params = [self.W, self.b]

    def _verbose(self, is_verbose, filter_shape, pool_size):
        if is_verbose:
            print('')
            print("Convolutional layer with {} kernels".format(filter_shape[-1]))
            print("------ Kernel size \t {} X {}".format(filter_shape[0], filter_shape[1]))
            print("------ Pooling size \t {} X {}".format(pool_size[1], pool_size[2]))
            print('')


class FullyConnectedLayer(BaseLayer):
    """
    Fully connected layer. Weights and input summed. 
    Result put through an activation function, which result in the layer output. 
    Dropout can also be applied depending on the drop variable.
    """

    def __init__(self, input, n_in, n_out, activation=tf.nn.relu
                 , W=None, b=None, name=None, dropout=False,
                 keep_prob=1.0, verbose=True):
        """
        :param input: input data
        :param n_in: Incoming units
        :param n_out: the nodes of this layer
        :param activation: activation function
        :param W: Supplied layer weights.
        :param b: Supplied biases.
        :param verbose: Print layer arch. in console.
        """
        super(FullyConnectedLayer, self).__init__(input)
        self._verbose(verbose, n_in, n_out, activation)

        W_bound = np.sqrt(6.0 / (n_in + n_out)) * 4

        self.set_weights_uniform(W, -W_bound, W_bound, (n_in, n_out), name=name)
        self.set_bias(b, n_out, name=name)

        lin_out = tf.matmul(input, self.W) + self.b

        fc_out = lin_out if activation is None else activation(lin_out)

        if dropout:
            self.output = tf.nn.dropout(fc_out, keep_prob=keep_prob)
        else:
            self.output = fc_out

        self.params = [self.W, self.b]

    def _verbose(self, is_verbose, n_in, n_out, activation):
        if is_verbose:
            print('')
            print("Fully connected layer with {} nodes".format(n_out))
            print("------ Incoming connections: {}".format(n_in))
            print("------ Activation is {}".format(activation))


class OutputLayer(BaseLayer):
    """
    Output layer for convolutional neural network. 
    Support many different loss functions
    """

    def __init__(self, input, n_in, n_out, activation=tf.nn.sigmoid,
                 batch_size=32, loss='crossentropy', out_patch=False,
                 W=None, b=None, name=None, verbose=True):
        super(OutputLayer, self).__init__(input)
        self._verbose(verbose, n_in, n_out, loss)

        if loss == "bootstrapping":
            print("bootstrapping")
            self.negative_log_likelihood = self.loss_bootstrapping
        elif loss == 'crosstrapping':
            print('crosstrapping')
            self.negative_log_likelihood = self.loss_crosstrapping
        elif loss == 'bootstrapping_soft':
            print('bootstrapping_soft')
            self.negative_log_likelihood = self.loss_bootstrapping_soft
        elif loss == 'bootstrapping_confident':
            print('bootstrapping_confident')
            self.negative_log_likelihood = self.loss_confident_bootstrapping
        elif loss == "bootstrapping_union":
            print("bootstrapping_union")
            self.negative_log_likelihood = self.loss_stochastic_union_bootstrapping
        else:
            print("crossentropy")
            self.negative_log_likelihood = self.loss_crossentropy

        if out_patch:
            W_bound = np.sqrt(6.0 / (n_in + n_out[0] * n_out[1])) * 4
            self.set_weights_uniform(
                W, -W_bound, W_bound, (n_in, n_out[0] * n_out[1]), name=name)
            self.set_bias(b, n_out[0] * n_out[1], name=name)
            self.output = tf.reshape(
                activation(tf.matmul(input, self.W) + self.b), [-1, n_out[0], n_out[1]])
            self.size = n_out[0] * n_out[1] * batch_size
        else:
            W_bound = np.sqrt(6.0 / (n_in + n_out)) * 4
            self.set_weights_uniform(W, -W_bound, W_bound, (n_in, n_out), name=name)
            self.set_bias(b, n_out, name=name)
            self.output = activation(tf.matmul(input, self.W) + self.b)
            self.size = n_out * batch_size

        self.params = [self.W, self.b]
        # self.output = T.clip(self.output, 1e-7, 1.0 - 1e-7)

    def loss_crossentropy(self, y, factor=None):
        # return -tf.reduce_sum(y * tf.log(self.output + 1e-10))
        return -tf.reduce_sum(y * tf.log(self.output + 1e-10) + (1 - y) * tf.log(1 - self.output + 1e-10)) / 256

    def loss_bootstrapping(self, y, factor=1):
        # Customized categorical cross entropy.
        p = self.output
        hard = tf.greater(p, 0.5)
        loss = (-tf.reduce_sum(factor * y + (1.0 - factor) * hard) * tf.log(p) -
                tf.reduce_sum(factor * (1.0 - y) + (1.0 - factor) *
                              (1.0 - hard)) * tf.log(1.0 - p))
        return loss / self.size

    def loss_bootstrapping_soft(self, y, factor=1):
        # Soft version of bootstrapping
        p = self.output
        loss = (-tf.reduce_sum(factor * y + (1.0 - factor) * p) * tf.log(p) -
                tf.reduce_sum(factor * (1.0 - y) + (1.0 - factor) *
                              (1.0 - p)) * tf.log(1.0 - p))
        return loss / self.size

    def loss_confident_bootstrapping(self, y, factor):
        # Only confident predictions are included.
        # Everything between 0.2 and 0.8 is disregarded. 60% of the range.
        p = self.output
        hard_upper = tf.greater(p, 0.8)
        hard_lower = tf.less(p, 0.2)
        loss = (-tf.reduce_sum(factor * y + (1.0 - factor) * hard_upper) * tf.log(p) -
                tf.reduce_sum(factor * (1.0 - y) + (1.0 - factor) *
                              hard_lower) * tf.log(1 - p))
        return loss / self.size

    def loss_crosstrapping(self, y, factor=1):
        # Almost the same as bootstrapping, except mean used for overall result.
        # More closely follows crossentropy implementation.
        # When factor is 1, crossentropy equals this implementation. So performance
        # without decreasing factor should be the same!
        p = self.output
        hard = tf.greater(p, 0.5)
        cross = -((factor * y * tf.log(p) + (1.0 - factor) * hard * tf.log(p)) +
                  factor * (1.0 - y) * tf.log(1.0 - p) + (1.0 - factor) * (1.0 - hard) * tf.log(1.0 - p))
        return tf.reduce_mean(cross)

    def loss_stochastic_union_bootstrapping(self):
        pass

    def mean_squared_error(self, y):
        # Returns the mean squared error.
        # Prediction - label squared, for all cells in all batches and pixels.
        return tf.reduce_mean(tf.pow(self.output - y, 2))

    def _verbose(self, is_verbose, n_in, n_out, loss):
        if is_verbose:
            print('')
            print("Output layer with {} outputs".format(n_out))
            print("------ Incoming connections: {}".format(n_in))
            print("------ Loss function : {}".format(loss))
            print('')


class AbstractModel(metaclass=ABCMeta):
    """
    Different architectures inherit from AbstractModel. 
    Contains methods needed by the Evaluator class.
    The abstract build method should be implemented by subclasses.
    """

    def __init__(self, params=None, verbose=None):  # Every layer appended to this variable.

        # layer 0= input, layer N = output
        self.layers = []
        self.output = None

        # self.L2_layers = []
        # self.input_data_dim = params.input_data_dim
        # self.output_labels_dim = params.output_label_dim
        # self.hidden = params.hidden_layer
        self.params = params

    def get_output_layer(self):
        assert len(self.layers) > 0
        return self.layers[-1]

    def get_cost(self, y, factor=1):
        """Get the loss function."""
        return self.get_output_layer().negative_log_likelihood(y, factor)

    def get_mean_squared_error(self, y):
        """Get the mean_squared_error"""
        return self.get_output_layer().mean_squared_error(y)

    def getL2(self):
        v = tf.Variable(tf.zeros(1))
        for layer in self.layers:
            temp = tf.reduce_sum(layer.W ** 2)
            new_v = tf.add(v, temp)
            v = tf.assign(v, new_v)
        return v

    def create_predict_function(self, x, keep_prob, data):
        pass

    @abstractclassmethod
    def build(self, x, keep_prob):
        return


class ConvModel(AbstractModel):
    """
    The build method dynamically creates the convolutional layers However the fully 
    connected layers are static.The final hidden layer is fully connected
    as well as the output layer. If there are init params, 
    the layers are initialized from these values. Otherwise,
    each layer's weights and biases are initialized by random values.
    """

    def __init__(self, verbose=True):
        super(ConvModel, self).__init__(verbose)
        # self.nr_kernels = params.nr_kernels
        # self.dropout_rates = params.dropout_rates
        # self.conv = params.conv_layers
        self.verbose = verbose
        # self.queue = deque([self.input_data_dim[0], -1])

    def build(self, x, keep_prob):
        print('Creating layers for convolutional neural network model')

        if self.verbose:
            print("------ Using supplied weights and bias.")

        conv1_layer = ConvPoolLayer(x, filter_shape=(13, 13, 3, 64),
                                    activation=tf.nn.relu, image_shape=(None, 64, 64, 3),
                                    padding="SAME", strides=(1, 4, 4, 1),
                                    pool_size=(1, 2, 2, 1), pool_strides=(1, 2, 2, 1),
                                    name="conv1")
        self.layers.append(conv1_layer)
        conv2_layer = ConvPoolLayer(conv1_layer.output, filter_shape=(4, 4, 64, 112),
                                    activation=tf.nn.relu, padding="SAME",
                                    strides=(1, 1, 1, 1), pool_size=(1, 1, 1, 1),
                                    pool_strides=(1, 1, 1, 1), dropout=True,
                                    keep_prob=keep_prob[0], name="conv2")
        self.layers.append(conv2_layer)
        conv3_layer = ConvPoolLayer(conv2_layer.output, filter_shape=(3, 3, 112, 80),
                                    activation=tf.nn.relu, padding="SAME",
                                    strides=(1, 1, 1, 1), pool_size=(1, 1, 1, 1),
                                    pool_strides=(1, 1, 1, 1), dropout=True,
                                    keep_prob=keep_prob[1], name="conv3")
        self.layers.append(conv3_layer)

        conv3_layer_out = conv3_layer.output
        conv3_layer_out_dim = conv3_layer_out.get_shape()
        dim_in = conv3_layer_out_dim[1] * conv3_layer_out_dim[2] * conv3_layer_out_dim[3]
        dim_in = dim_in.value
        flatten = tf.reshape(conv3_layer_out, [-1, dim_in])

        fc1_layer = FullyConnectedLayer(flatten, n_in=dim_in, n_out=4096,
                                        activation=tf.nn.relu, dropout=True,
                                        keep_prob=keep_prob[2], name="fc1")
        self.layers.append(fc1_layer)
        output_layer = OutputLayer(fc1_layer.output, n_in=4096, n_out=(16, 16),
                                   activation=tf.nn.sigmoid, out_patch=True)
        self.layers.append(output_layer)

        self.output = output_layer.output


class Evaluator(object):
    """
    The evaluator class contains the main training loop. 
    It receives the model and dataset and conducts the optimization.
    The number of epochs are supplied in run method, 
    while the main loop draws parameters directly from the loaded
    config.py. The learning rate, bootstrapping factor,
    curriculum switch and early stopping are set the loop inside
    """

    def __init__(self, model=None, dataset=None,
                 params=None, init_model=None, train=True):
        self.model = model
        self.data = dataset
        self.sess = init_model
        self.params = params
        self.report = {}
        self.events = []
        self.train = train

    def fit(self, epoch=1000, batch_size=16):
        self.nr_train_batches = self.data.get_total_number_of_batches(batch_size)
        self.nr_valid_batches = self._get_number_of_batches("valid", batch_size)
        self.nr_test_batches = self._get_number_of_batches("test_PyQt5", batch_size)
        self._train(max_epoch=epoch, batch_size=batch_size, init_model=self.sess, train=self.train)

    def simple_predict(self, sess, model_out, data, x,
                       keep_prob, keep_prob_value):
        print("Predicting simple......")
        result = sess.run(model_out, feed_dict={x: data,
                                                keep_prob: keep_prob_value,
                                                })
        return result

    def batch_predict(self, sess, model_out, data, x, y, batch_size,
                      keep_prob, keep_prob_value):
        print("Predicting batch......")
        nr_elements = data.shape[0]
        result = np.empty((nr_elements, y[0], y[1]), dtype="float32")
        batches = [data[x:x + batch_size] for x in range(0, nr_elements, batch_size)]
        idx = 0
        for batch in batches:
            res = sess.run(model_out, feed_dict={x: batch,
                                                 keep_prob: keep_prob_value})
            result[idx:idx + batch_size] = res
            idx += batch_size
        return result

    def _train(self, max_epoch, batch_size, init_model=None, train=True):
        print("Building model")

        x = tf.placeholder("float32", [None, 64, 64, 3])
        y = tf.placeholder("float32", [None, 16, 16])
        keep_prob = tf.placeholder("float32", [3])
        mix_factor = tf.placeholder("float32")
        learning_rate = tf.placeholder("float32")

        self.model.build(x, keep_prob)
        model_out = self.model.output
        self.output = model_out

        cost = self.model.get_cost(y, mix_factor)  # + self.model.getL2() * self.params.l2_reg

        optimizer = tf.train.MomentumOptimizer(learning_rate, self.params.momentum)

        train_step = optimizer.minimize(cost)

        init_op = tf.global_variables_initializer()
        print("Model built!!")
        print('')

        print('Training model')
        keep_prob_value = self.params.keep_prob
        keep_prob_valid_test = self.params.keep_prob_valid_test

        patience = self.params.initial_patience  # look as this many examples regardless
        patience_increase = self.params.patience_increase  # wait this much longer when a new best is found
        improvement_threshold = self.params.improvement_threshold  # a relative improvement of this much is considered significant

        learning_rate_value = self.params.learning_rate
        learning_adjustment = self.params.learning_adjustment
        learning_decrease = self.params.learning_decrease
        nr_learning_adjustments = 0
        print("------ Initial learning rate {}".format(learning_rate_value))

        max_factor = self.params.factor_rate
        factor_adjustment = self.params.factor_adjustment
        factor_decrease = self.params.factor_decrease
        factor_minimum = self.params.factor_minimum
        print("------ Initial loss mixture ratio {}".format(max_factor))

        curriculum = self.params.curriculum_enable
        curriculum_start = self.params.curriculum_start
        curriculum_adjustment = self.params.curriculum_adjustment

        # gui_frequency = 500
        validation_frequency = min(self.nr_train_batches, patience / 2)
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0

        nr_chunks = self.data.get_chunk_number()
        epoch = 0
        done_looping = False
        nr_iter = 0

        saver = tf.train.Saver()

        with tf.Session() as sess:
            if init_model is None or train:
                sess.run(init_op)

                if train and init_model is not None:
                    sess = saver.restore(train, init_model)

                chunk_batches = self.data.get_elements(0) / batch_size
                validation_score = self._run_get_loss(sess, cost,
                                                      batch_size, x, y,
                                                      keep_prob, keep_prob_valid_test, "valid")
                test_score = self._run_get_loss(sess, cost,
                                                batch_size, x, y,
                                                keep_prob, keep_prob_valid_test, "test_PyQt5")

                try:
                    while epoch < max_epoch and not done_looping:
                        epoch += 1

                        # TODO ：adjust learning rate
                        if (epoch % learning_adjustment == 0):
                            learning_rate_value *= learning_decrease
                            nr_learning_adjustments += 1

                            learning_adjustment = max(10, int(learning_adjustment / 2))
                            print("------ New learning rate {}".format(learning_rate_value))

                        # TODO: adjust max_factor
                        if epoch > factor_adjustment:
                            max_factor = max(max_factor * factor_decrease, factor_minimum)
                            print("------ New convex combination {}".format(max_factor))

                        # TODO: save model
                        if epoch % 20 == 0:
                            print("------ Store temp model")
                            # TODO:file_path
                            save_path = r"./road_model/model_{}/model.ckpt".format(str(epoch))
                            saver.save(sess, save_path=save_path)

                        # TODO: transform stage about curriculum
                        if curriculum and epoch % curriculum_adjustment == 0 and epoch >= curriculum_start:
                            print("------ Mix examples from next stage with training data")
                            self.data.mix_in_next_stage()

                        # TODO: training model
                        for chunk_index in range(nr_chunks):
                            self.data.switch_active_training_set(chunk_index)
                            nr_elements = self.data.get_elements(chunk_index)
                            train_data = self.data.data_set["train"]
                            batches = [[train_data[0][x:x + batch_size], train_data[1][x:x + batch_size]]
                                       for x in range(0, nr_elements, batch_size)]

                            for batch in batches:
                                loss = self._run_train(sess, [cost, train_step], batch, x, y, keep_prob,
                                                       keep_prob_value,
                                                       mix_factor, max_factor, learning_rate, learning_rate_value)

                                # TODO 每隔多少个batch打印一次信息
                                if nr_iter % 2 == 0:
                                    print("------ Training @ iter = {}. Patience = {}. Loss = {}".format(
                                        nr_iter, patience, loss))

                                # TODO ==== EVAULATE ====
                                if (nr_iter + 1) % validation_frequency == 0:
                                    validation_score = self._run_get_loss(sess, cost,
                                                                          batch_size, x, y,
                                                                          keep_prob, keep_prob_valid_test, "valid")
                                    test_score = self._run_get_loss(sess, cost,
                                                                    batch_size, x, y,
                                                                    keep_prob, keep_prob_valid_test, "test_PyQt5")
                                    self.events.append({
                                        "epoch": epoch,
                                        "training_loss": loss,
                                        "validation_loss": validation_score,
                                        "test_loss": test_score,
                                        "training_rate": learning_rate_value})

                                    # TODO ==== EARLY STOPPING ====
                                    if validation_score < best_validation_loss:
                                        # improve patience if loss improvement is good enough
                                        if validation_score < best_validation_loss * improvement_threshold:
                                            patience = max(patience, nr_iter * patience_increase)
                                            print(
                                                "------ New best validation loss.Patience increased to {}".format(
                                                    patience))

                                        # save best validation score and iteration number
                                        best_validation_loss = validation_score
                                        best_iter = nr_iter

                                if patience < nr_iter:
                                    done_looping = True
                                    break
                                nr_iter += 1

                except KeyboardInterrupt:
                    self.set_result(best_iter, nr_iter, validation_score, test_score,
                                    nr_learning_adjustments, epoch)
                self.set_result(best_iter, nr_iter, validation_score, test_score,
                                nr_learning_adjustments, epoch)

                self.sess = sess
            else:
                saver.restore(sess, init_model)

    def _run_get_loss(self, sess, cost, batch_size, x, y,
                      keep_prob, keep_prob_value, name):

        temp_data = self.data.data_set[name]
        nr_elements = temp_data[0].shape[0]
        temp_batches = [[temp_data[0][x:x + batch_size], temp_data[1][x:x + batch_size]]
                        for x in range(0, nr_elements, batch_size)]
        result = []
        for batch in temp_batches:
            temp = sess.run(cost, feed_dict={x: batch[0],
                                             y: batch[1],
                                             keep_prob: keep_prob_value,
                                             })
            result.append(temp)
        return np.mean(np.array(result))

    def _run_train(self, sess, cost, batch_data, x, y,
                   keep_prob, keep_prob_value,
                   mix_factor, mix_factor_value,
                   learning_rate, learning_rate_value):
        data, label = batch_data
        result = sess.run(cost, feed_dict={x: data,
                                           y: label,
                                           keep_prob: keep_prob_value,
                                           mix_factor: mix_factor_value,
                                           learning_rate: learning_rate_value
                                           })
        return result[0]

    def _get_number_of_batches(self, data_set_name, batch_size):
        set_x, set_y = self.data.data_set[data_set_name]
        nr_of_batches = set_x.shape[0]
        nr_of_batches /= batch_size
        return int(nr_of_batches)

    def set_result(self, best_iter, n_iter, valid, test,
                   nr_learning_adjustments, epoch):
        valid_end_score = valid
        test_end_score = test
        print("Optimization completed")
        print('Best validation score of %f obtained at iteration %i, '
              'with test_PyQt5 performance %f' %
              (valid_end_score, best_iter + 1, test_end_score))
        self.report['evaluation'] = {
            'best_iteration': best_iter + 1, 'iteration': iter, 'test_score': test_end_score,
            'valid_score': valid_end_score,
            'learning_adjustments': nr_learning_adjustments, 'epoch': epoch
        }
        self.report['dataset'] = self.data.get_report()

    def get_report(self):
        return self.report


if __name__ == '__main__':
    # data = np.zeros((100, 64, 64, 3), dtype="float32")
    # conv = ConvPoolLayer(data, [2, 2, 3, 64], image_shape=(10, 64, 64, 3),
    #                      dropout=True, name="conv1")  # print(conv.input)
    # print(conv.input.shape)
    # print(conv.W)
    # print(conv.b)
    # print(conv.output)
    #
    # data1 = np.zeros((100, 100), dtype="float32")
    # fc = FullyConnectedLayer(data1, 100, 30, name="FC1")
    # print(fc.input.shape)
    # print(fc.W)
    # print(fc.b)
    # print(fc.output)
    #
    # data2 = np.zeros((100, 100), dtype="float32")
    # out = OutputLayer(data2, 100, 10, name="output")
    # print(out.input.shape)
    # print(out.output)
    # print(out.W)
    # print(out.b)
    #
    # data3 = np.zeros((100, 100), dtype="float32")
    # out2 = OutputLayer(data3, 100, (10, 10), out_patch=True, name="output2")
    # print(out2.input.shape)
    # print(out2.output)
    # print(out2.W)
    # print(out2.b)

    # x = tf.placeholder("float32", [None, 64, 64, 3])
    # y_ = tf.placeholder("float32", [None, 16, 16])
    # keep_prob = tf.placeholder("float32", [3])
    #
    # model = ConvModel()
    # y = model.build(x, keep_prob)
    # print(model.built)
    dataset = AerialDataset()
    path = r"/media/sunwl/sunwl/datum/roadDetect_project/Massachusetts/"
    params = dataset_params
    dataset.load(path, params=params)
    dataset.switch_active_training_set(0)
    train_data = dataset.data_set['train']
    print("train")
    print(train_data[0].shape)
    print(train_data[1].shape)
    valid_data = dataset.data_set['valid']
    print("valid")
    print(valid_data[0].shape)
    print(valid_data[1].shape)
    test_data = dataset.data_set['test_PyQt5']
    print("test_PyQt5")
    print(test_data[0].shape)
    print(test_data[1].shape)
    print(dataset.get_report())
    dataset.switch_active_training_set(2)
    dataset.get_elements(2)
    new_data = dataset.data_set['train']
    print(new_data[0].shape)

    model = ConvModel()

    evaluator = Evaluator(model, dataset, optimization_params)
    evaluator.fit(epoch=10, batch_size=16)
