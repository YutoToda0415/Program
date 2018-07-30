# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 18:32:56 2017

@author: Yuto
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#training set data 
x = tf.placeholder(tf.float32,[None,5])#p051
w = tf.Variable(tf.zeros([5,1]))#p053

# y = x * w;
y = tf.matmul(x,w)

t = tf.placeholder(tf.float32,[None,1])#tは12X1行列に相当する
loss = tf.reduce_sum(tf.square(y-t))# 誤差関数 P.50 54

train_step = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())


#トレーニングセットを用意する
#train_t = np.array([5.2,5.7,8.6])


