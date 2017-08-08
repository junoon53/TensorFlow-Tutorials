#!/usr/bin/env python

import tensorflow as tf
import numpy as np

trX = []
trY = []

trX1 = np.linspace(0, 10, 1000)
trX2 = np.linspace(-1,1,1000)
trX3 = np.linspace(0,5,1000)

trX = np.zeros([1000,3])
for i in range(1000):
    trX[i,:] = [trX1[i],trX2[i],trX3[i]]

w_true = np.array([2,3,0.64])

trY = np.sum(w_true*trX,axis=1) #+ np.random.randn()*0.33

# print(trX,trY,sep="\n")
print(trX.shape)
print(trY.shape)

Y = tf.placeholder(tf.float32, name="Y")
X = tf.placeholder(tf.float32, shape=(3,), name="X")
w = tf.Variable([0.1,0.1,0.1], name="weights")

loss = tf.square(Y - tf.reduce_sum(tf.multiply(X, w)))

init_op = tf.global_variables_initializer()
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:

    sess.run(init_op)
    
    for i in range(100):
        for (x,y) in zip(trX,trY):
            # print(x,y,sep=" : ")
            sess.run(train_op, feed_dict={X: x, Y: y})
    print(sess.run(w))

    error = 0.0
    for (x,y) in zip(trX,trY):
        error += abs(y - np.sum(x*sess.run(w))) 
    error = error/1000
    print(error*100)


