'''
Convolutional Neural Network
@author: kareem arab

refs //
    - ...
'''
import numpy as np
import tensorflow as tf

class CONVNetwork(object):

    def __init__(self, data, sess, epochs, batch_size):
        self.trX, self.trY, self.vaX, self.vaY = data
        self.sess = sess
        self.epochs = epochs
        self.batch_size = batch_size
        # self.train_writer = train_writer

        self.test_size = 256
        self.epoch_accuracies = []

        self.X = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1], name='X')
        self.Y = tf.compat.v1.placeholder(tf.float32, [None, 10], name='Y')
        self.training = tf.compat.v1.placeholder_with_default(False, (), name='mode')

        self.y_pred, self.logits_ = self.graph(self.X, tr_bool=self.training)

        self.count = tf.equal(tf.argmax(self.Y, axis=1), tf.argmax(self.y_pred, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(self.count, tf.float32), name='accuracy')

        tf.summary.histogram("accuracy", self.accuracy)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.Y))

        self.train_op = tf.compat.v1.train.RMSPropOptimizer(0.001, 0.9).minimize(self.cost)
        self.predict_op = tf.argmax(self.y_pred, 1)

        self.saver = tf.compat.v1.train.Saver()

    def init_weights(self, shape):
        return tf.Variable(tf.random.normal(shape, stddev=0.01))

    def graph(self, X, tr_bool=False):
        '''
        primary computational graph.

        reqs //
            - topology
                - [conv2d->pooling]x2 -> fc -> softmax output
                - feature-maps // 32 -> 64 -> 128 -> 10
        '''
        with tf.compat.v1.variable_scope('graph', reuse=tf.compat.v1.AUTO_REUSE):
            conv_layer_1 = tf.layers.conv2d(X, filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
            pool_layer_1 = tf.layers.max_pooling2d(conv_layer_1, pool_size=[2, 2], strides=2)

            conv_layer_2 = tf.layers.conv2d(pool_layer_1, filters=64, kernel_size=[3, 3],padding='same', activation=tf.nn.relu)
            pool_layer_2 = tf.layers.max_pooling2d(conv_layer_2, pool_size=[2, 2], strides=2)

            shape = pool_layer_2.get_shape().as_list()
            dense_layer = tf.reshape(pool_layer_2, [-1, np.prod(shape[1:])])

            dense1 = tf.layers.dense(dense_layer, units=128, activation=tf.nn.relu)
            dense1 = tf.layers.dropout(dense1, rate=0.25, training=tr_bool)
            ol = tf.layers.dense(dense1, units=10, name='ol')

            return tf.nn.softmax(ol), ol

    def train(self):
        self.sess.run(tf.global_variables_initializer())

        for i in range(self.epochs):
            ind = np.arange(self.trX.shape[0])
            np.random.shuffle(ind)
            self.trX = self.trX[ind]
            self.trY = self.trY[ind]
            # merge = tf.summary.merge_all()
            training_batch = zip(range(0, len(self.trX), self.batch_size), range(self.batch_size, len(self.trX)+1, self.batch_size))
            for counter, (start, end) in enumerate(training_batch):

                self.sess.run([self.train_op], feed_dict={
                    self.X: self.trX[start:end],
                    self.Y: self.trY[start:end],
                    self.training: True
                })
            # self.train_writer.add_summary(summary, i)

            acc = self.get_acc(self.vaX, self.vaY)
            self.epoch_accuracies.append(acc)

    def get_acc(self, images, labels):
        '''
        test accuracy between a set of images and labels
        '''
        print('testing accuracy...')
        accuracy_agg = []
        batch = zip(range(0, len(images), 32), range(32, len(images) + 1, 32))
        for i, (start, end) in enumerate(batch):
            accuracy = self.sess.run(self.accuracy, feed_dict={
                self.X: images[start:end],
                self.Y: labels[start:end]
            })
            accuracy_agg.append(accuracy)

        acc = sum(accuracy_agg) / len(accuracy_agg)
        print('acc //', acc)

        return acc

    def run_y_pred(self, images):
        '''
        get output from graph self.y_pred
        '''
        print('running images and labels on y_pred...')
        result_agg = np.empty((128, 10))

        batch = zip(range(0, len(images), 32), range(32, len(images) + 1, 32))
        for i, (start, end) in enumerate(batch):

            result = self.sess.run(self.y_pred, feed_dict={
                self.X: images[start:end]
            })
            result_agg[start:end] = result

        return result_agg