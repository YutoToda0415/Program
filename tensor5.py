# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:50:15 2018

@author: Yuto
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.examples.tutorials.mnist import input_data


np.random.seed(20180122)

mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

x = tf.placeholder(tf.float32, [None,784])
w = tf.Variable(tf.zeros([784,10]))
w0 = tf.Variable(tf.zeros([10]))
f = tf.matmul(x,w) + w0
p = tf.nn.softmax(f)

t = tf.placeholder(tf.float32, [None,10])
loss = -tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdamOptimizer().minimize(loss)

correct_prediction = tf.equal(tf.argmax(p,1),tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

 
