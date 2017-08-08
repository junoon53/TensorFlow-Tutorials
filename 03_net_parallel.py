#!/usr/bin/env python

import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from multiprocessing import Process


def run_in_parallel(in_dirpath,out_dirpath,target,args=None):
    if in_dirpath:
        fnames = os.listdir(in_dirpath)
    else:
        fnames = args
        
    if out_dirpath:
        if not os.path.exists(out_dirpath):
            os.makedirs(out_dirpath)
    procs = []
    for i in xrange(len(fnames)):
        p = Process(target=target,args=(i,in_dirpath,fnames[i],out_dirpath,args))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

def train(i,in_dirpath,hidden_dim,out_dirpath,args):

    print("hidden_dim",hidden_dim)
    print("using device",i)

    with tf.device('/gpu:%d'%(i)):
        w_h = init_weights([784, hidden_dim]) # create symbolic variables
        w_o = init_weights([hidden_dim, 10])

    py_x = model(X, w_h, w_o)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
    predict_op = tf.argmax(py_x, axis=1)

# Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()

        for i in range(100):
            for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        print(i,"train accuracy =", np.mean(np.argmax(trY, axis=1) == sess.run(predict_op, feed_dict={X: trX})))
        print(i,"test accuracy =", np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX})))


if __name__ == "__main__":

    hidden_dims = [50,100,200,400]#,625,800,1000]
    run_in_parallel(None,None,train,hidden_dims)

