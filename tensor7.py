# -*- coding: utf-8 -*-
"""
Created on Thu May 24 23:03:47 2018

@author: Yuto
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy.random import *
import pandas as pd
from pandas import DataFrame, Series

np.random.seed(20180525)
tf.set_random_seed(20180525)

num_units1 = 2
num_units2 = 2

x = tf.placeholder(tf.float32, [None,2])
t = tf.placeholder(tf.float32, [None,1])


w1 = tf.Variable(tf.truncated_normal([2,num_units1]))
b1 = tf.Variable(tf.zeros([num_units1]))
hidden1 = tf.nn.tanh(tf.matmul(x,w1) +b1 )

w2 = tf.Variable(tf.truncated_normal([num_units1,num_units2]))
b2 = tf.Variable(tf.zeros([num_units2]))
hidden2 = tf.nn.tanh(tf.matmul(hidden1,w2) + b2)

w0 = tf.Variable(tf.zeros([num_units2,1]))
b0 = tf.Variable(tf.zeros([1]))
p = tf.nn.sigmoid(tf.matmul(hidden2,w0) + b0)

xxx= np.array([1,1,1,0,0,1,0,0])
xxx = xxx.reshape([4,2])

ttt = np.array([1,0,0,1])
ttt = ttt.reshape([4,1])


step_num = 100
test_x = np.zeros([step_num,2])
test_t = np.zeros([step_num,1])

i = 0
for _ in range(step_num):
    test_x[i,0] = randint(100) % 2
    test_x[i,1] = randint(123) % 2
          
    if test_x[i,0] == test_x[i,1]:
        test_t[i] = 1
    else:
        test_t[i] = 0
    print('test_x : %d %d  test_t : %d' %(test_x[i,0] ,test_x[i,1] , test_t[i]))              
    i += 1
                      

correct_prediction = tf.equal(p,t)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())
    
i = 0 
for _ in range(200):
    i += 1
    batch_xs = xxx
    batch_ts = ttt
    sess.run(p, feed_dict={x: batch_xs, t:batch_ts})
    
    for _ in range(20):
        acc_val = sess.run(accuracy, feed_dict={x: test_x, t:test_t})
        print ('Step: %d, Accuraccy: %f' % (i, acc_val))
    
print(t)
