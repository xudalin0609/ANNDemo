# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


class ANN:

    def __init__(self, size, learning_rate=0.00001, log_path):
        self.learning_rate = learning_rate
        self.size = size
        self.log_path = log_path

    def defineANN(self):
        prevSize = self.input.shape[0].value
        prevOut = None
        size = self.size
        layer = 1
        for currentSize in size[:-1]:
            weight = tf.Variable(tf.truncated_normal([prevSize, currentSize], stddev=1/np.sqrt(float(prevSize))))
            tf.summary.histogram('hidden layer%d' % (layer), weight)
            biases = tf.Variable(tf.zeros([currentSize]))
            # 激活层使用sigmod
            prevOut = tf.sigmoid(tf.matmul(prevOut, weight) + biases)
            prevSize = currentSize
        weight = tf.Variable(tf.truncated_normal([prevSize, size[-1]], stddev=1/np.sqrt(float(prevSize))))
        tf.summary.histogram('out layer:', weight)
        biases = tf.Variable(tf.zeros([size[-1]]))
        self.out = tf.matmul(prevOut, weight) + biases
        return self

    def defineLoss(self):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(self.out, self.label, name='loss')
        self.loss = tf.reduce_mean(loss, name='average_loss')
        return self

    def SGD(self, X, y, miniBatchFraction, epoch):
        tf.summary.scalar('loss', self.loss)
        summary = tf.summary.merge_all()
        method = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        optimizer = method.minimize(self.loss)
        batchSize = int(X.shape[0] * miniBatchFraction)
        batchNum = int(np.ceil(1/miniBatchFraction))
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
        step = 0
        while step < epoch:
            for i in range(batchNum):
                batchX = X[i*batchSize: (i+1)*batchSize]
                batchY = y[i*batchSize: (i+1)*batchSize]
                sess.run([optimizer],
                         feed_dict={self.input: batchX, self.label: batchY})
            step += 1
            summary_str = sess.run(summary, feed_dict={self.input: X, self.label: y})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
        self.sess = sess
        return self

    def fit(self, X, y, miniBatchFraction, epoch):
        self.input = tf.placeholder(tf.float32, shape=[None, X.shape[1]], name='X')
        self.label = tf.placeholder(tf.int8, shape=[None, self.size[-1]], name='y')
        self.defineANN()
        self.defineLoss()
        self.SGD(X, y, miniBatchFraction, epoch)

    def predict(self, pred_X):
        sess = self.sess
        outLayer = tf.nn.softmax(logits=self.out, name='outLayer')
        pred_y = sess.run(outLayer, feed_dict={self.input: pred_X})
        return pred_y