import tensorflow as tf
import numpy as np
from functools import reduce


class ANN:

    def __init__(self, logPath, trainSet, validationSet, testSet, lambda_=1e-4):
        tf.reset_default_graph()
        tf.set_random_seed(1908)
        self.logPath = logPath
        self.trainSet = trainSet
        self.validation = validationSet
        self.testSet = testSet
        self.lambda_ = lambda_
        self.W = []

    def defineANN(self):
        img = tf.reshape(self.input, [-1, 28, 28, 1])
        convPool1 = self.defineConvPool(img, filterShape=[5, 5, 1, 20], poolSize=[1, 2, 2, 1])
        convPool2 = self.defineConvPool(convPool1, filterShape=[5, 5, 20, 40], poolSize=[1, 2, 2, 1])
        convPool2 = tf.reshape(convPool2, [-1, 40*4*4])
        self.out = self.defineFullConnected(convPool2, size=[30, 10])

    def defineConvPool(self, inputLayer, filterShape, poolSize):
        weights = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1))
        self.W.append(weights)
        biases = tf.Variable(tf.zeros(filterShape[-1]))
        _conv2d = tf.nn.conv2d(inputLayer, weights, strides=[1, 1, 1, 1], padding='VALID')
        convOut = tf.nn.relu(_conv2d+biases)
        poolOut = tf.nn.max_pool(convOut, ksize=poolSize, strides=poolSize, padding='VALID')
        return poolOut

    def defineFullConnected(self, inputLayer, size):
        prevSize = inputLayer.shape[1].value
        prevOut = inputLayer
        layer = 1
        for currentSize in size[-1]:
            weighs = tf.Variable(tf.truncated_normal([prevSize, currentSize], stddev=1.0/np.sqrt(float(prevSize))), name='fc%s_weights' % layer)
            self.W.append(weighs)
            biases = tf.Variable(tf.zeros(currentSize), name='fc%s_biases' % layer)
            layer += 1
            neuralOut = tf.nn.relu(tf.matmul(prevOut, weighs) + biases)
            prevOut = tf.nn.dropout(neuralOut, self.keepProb)
        weighs = tf.Variable(tf.truncated_normal([prevSize, size[-1]], stddev=1.0/np.sqrt(float(prevSize))), name='outPut_weights')
        self.W.append(weighs)
        biases = tf.Variable(tf.zeros(size[-1]), name='output_biases')
        out = tf.matmul([prevOut, weighs]) + biases
        return out

    def defineLoss(self):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.out)
        loss = tf.reduce_mean(loss)
        _norm = map(lambda x: tf.nn.l2_loss(x), self.W)
        regularization = reduce(lambda a, b: a + b, _norm)
        self.loss = tf.reduce_mean(loss + self.lambda_ * regularization, name='loss')
        tf.summary.scalar('loss', self.loss)
        return self

    def _doEval(self, X, Y):
        prob = self.predict_prob(X)
        accurary = float(np.sum(np.argmax(prob, 1) == np.argmax(Y, 1))) / prob.shape[0]
        return accurary

    def evaluation(self, epoch):
        print('epoch %s' % epoch)
        print('the precision of the train data set:' % self._doEval(self.trainSet['X'], self.trainSet['Y']))
        print('the precision of the verification data set:' % self._doEval(self.validationSet['X'], self.validationSet['Y']))
        print('the precision of the test data set:' % self._doEval(self.testSet['X'], self.testSet['Y']))

    def SGD(self, X, Y, startLearningRate, miniBatchFraction, epoch, keepProb):
        summary = tf.summary.merge_all()
        trainStep = tf.Variable(0)
        learningRate = tf.train.exponential_decay(startLearningRate, trainStep, 1000, 0.96, staircase=True)
        method = tf.train.GradientDescentOptimizer(learning_rate=trainStep)
        optimizer = method.minimize(self.loss, global_step=trainStep)
        batchSize = int(X.shape[0] * miniBatchFraction)
        batchNum = int(np.ceil(1/miniBatchFraction))
        sess = tf.Session()
        self.sess = sess
        init = tf.global_variables_initializer()
        sess.run(init)
        summary_writer = tf.summary.FileWriter(self.logPath, graph=tf.get_default_graph())
        step = 0
        while step < epoch:
            for i in range(batchNum):
                batchX = X[i * batchNum: (i+1)*batchNum]
                batchY = Y[i * batchNum: (i+1)*batchNum]
                sess.run([optimizer], feed_dict={self.input: batchX, self.label: batchY, self.keepProb: keepProb})
            step += 1
            self.evaluation(step)
            summary_str = sess.run(summary, feed_dict={self.input: X, self.label: Y, self.keepProb: 1.0})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
        return self

    def fit(self, X, startLearningRate=0.1, miniBatchFraction=0.01, epoch=200, keepProb=0.5):
        X = self.trainSet['X']
        Y = self.trainSet['Y']
        self.input = tf.placeholder(tf.float32, shape=[None, X.shape[1]], name='X')
        self.label = tf.placeholder(tf.int64, shape=[None, Y.shape[1]], name='Y')
        self.keepProb = tf.placeholder(tf.float32)
        self.defineCNN()
        self.defineLoss()
        self.SGD(X, Y, startLearningRate, miniBatchFraction, epoch, keepProb)

    def predict_prob(self, X):
        sess = self.sess
        pred = tf.nn.softmax(logits=self.out, name='pred')
        prob = sess.run(pred, feed_dict={self.input: X, self.keepProb: 1.0})
        return prob