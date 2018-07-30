# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 14:30:27 2017

@author: Yuto
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

b_train = -1
w_train = 0.7
x_train = np.random.random((1,100))
y_train = x_train * w_train + b_train +1.0 * np.random.randn(1,100)

plt.figure(1)
plt.plot(x_train, y_train, 'ro', label='Data')

# 変数の定義
x = tf.placeholder(tf.float32,name = "input")
y = tf.placeholder(tf.float32,name = "output")
w = tf.Variable(np.random.randn(),name = "weight")
b = tf.Variable(np.random.randn(),name = "bias")

# 線形回帰のモデル
y_pred = tf.add(tf.multiply(x,w),b)

# 損失関数
loss = tf.reduce_mean(tf.pow(y_pred - y, 2))

# Optimizer
# 勾配降下法
learning_rate = 0.01
momentum_rate = 0.1
learning_rate1 =0.5
learning_rate2 =0.025
train_ops =[ tf.train.MomentumOptimizer(learning_rate, momentum_rate).minimize(loss) ,
             tf.train.AdagradOptimizer(learning_rate2).minimize(loss),
             tf.train.AdadeltaOptimizer(learning_rate1).minimize(loss),
             tf.train.AdamOptimizer().minimize(loss),
             tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)]

w_output = np.zeros([len(train_ops),1])
b_output = np.zeros([len(train_ops),1])



# セッション
for i in range(len(train_ops)):
    
    with tf.Session() as sess:
    #init_op = tf.initialize_all_variables()
    #sess.run(init_op)
  # 変数の初期化
        training_step = 5000
        validation = 10

        sess.run(tf.global_variables_initializer())
        step = 0
        for step in range(training_step):
            sess.run(train_ops[i],feed_dict={x:x_train, y:y_train})
        
        #途中経過
            if step % validation == 0:
                loss_output = sess.run(loss, feed_dict={x:x_train, y:y_train})
                w_output[i] = sess.run(w)#これがとても大事なところ
                b_output[i] = sess.run(b)#これがとても大事なところ
            
for i in range(len(train_ops)):
    
    x_line = np.linspace(0,1.0,100)
    out = x_line * w_output[i] + b_output[i]

    plt.plot(x_line,out,'b-')
    
plt.show()



