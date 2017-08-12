#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

def model(X, p_keep_input, p_keep_hidden): # this network is the same as the previous one except with an extra hidden layer + dropout

    w_h1 = tf.get_variable("w_h1",shape=[784,625]) # tensorflow inits these variables automatically when tf.global_variables_initializer() is run
    b_h1 = tf.get_variable("b_h1",shape=[625])
    w_h2 = tf.get_variable("w_h2",shape=[625,625])
    b_h2 = tf.get_variable("b_h2",shape=[625])
    w_o = tf.get_variable("w_o",shape=[625,10])
    
    X = tf.nn.dropout(X, p_keep_input)

    h = tf.nn.relu(tf.matmul(X, w_h1) + b_h1)
    h = tf.nn.dropout(h, p_keep_hidden)

    h2 = tf.nn.relu(tf.matmul(h, w_h2) + b_h2)
    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)


X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])
p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

py_x = model(X, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.85).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                        p_keep_input: 1.0, p_keep_hidden: 0.5})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                        sess.run(predict_op, feed_dict={X: teX, 
                                                        p_keep_input: 1.0,
                                                        p_keep_hidden: 1.0})))
