# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 09:50:50 2018

@author: Yuto
"""

import time
import numpy as np
import tensorflow as tf
 
# ネットワークの定義
 
in_size = 4
hidden_size = 20
out_size = 3
# プレースホルダー
x_ = tf.placeholder(tf.float32, shape=[None, in_size])
y_ = tf.placeholder(tf.float32, shape=[None, out_size])
# 順伝播のネットワークを作成
fc1_w = tf.Variable(tf.truncated_normal([in_size, hidden_size], stddev=0.1), dtype=tf.float32) # 入力層の重み
fc1_b = tf.Variable(tf.constant(0.1, shape=[hidden_size]), dtype=tf.float32) # 入力層のバイアス
fc1 = tf.nn.sigmoid(tf.matmul(x_, fc1_w) + fc1_b) # 全結合
fc2_w = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], stddev=0.1), dtype=tf.float32) # 隠れ層の重み
fc2_b = tf.Variable(tf.constant(0.1, shape=[hidden_size]), dtype=tf.float32) # 隠れ層のバイアス
fc2 = tf.nn.sigmoid(tf.matmul(fc1, fc2_w) + fc2_b) # 全結合
fc3_w = tf.Variable(tf.truncated_normal([hidden_size, out_size], stddev=0.1), dtype=tf.float32) # 出力層の重み
fc3_b = tf.Variable(tf.constant(0.1, shape=[out_size]), dtype=tf.float32) # 出力層のバイアス
y_pre = tf.matmul(fc2, fc3_w) + fc3_b # 全結合
# クロスエントロピー誤差
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_pre))
# 勾配法
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
# 正解率の計算
correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
# 学習
 
EPOCH_NUM = 100
BATCH_SIZE = 20
 
# データ
N = 100
in_size = 4
out_size = 3
iris = datasets.load_iris()
data = pd.DataFrame(data= np.c_[iris["data"], iris["target"]], columns= iris["feature_names"] + ["target"])
data = np.array(data.values)
perm = np.random.permutation(len(data))
data = data[perm]
train, test = np.split(data, [N])
train_x, train_y, test_x, test_y = [], [], [], []
for t in train:
    train_x.append(t[0:4])
    train_y.append(np.eye(out_size)[int(t[4])])
for t in test:
    test_x.append(t[0:4])
    test_y.append(np.eye(out_size)[int(t[4])])
train_x = np.array(train_x, dtype="float32")
train_y = np.array(train_y, dtype="float32")
test_x = np.array(test_x, dtype="float32")
test_y = np.array(test_y, dtype="float32")
 
# 学習
print("Train")
with tf.Session() as sess:
    st = time.time()
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCH_NUM):
        perm = np.random.permutation(N)
        total_loss = 0
        for i in range(0, N, BATCH_SIZE):
            batch_x = train_x[perm[i:i+BATCH_SIZE]]
            batch_y = train_y[perm[i:i+BATCH_SIZE]]
            total_loss += cross_entropy.eval(feed_dict={x_: batch_x, y_: batch_y})
            train_step.run(feed_dict={x_: batch_x, y_: batch_y})
            total_accuracy += accuracy.eval(feed_dict={x_: batch_x, y_: batch_y})
        acc = accuracy.eval(feed_dict={x_: train_x, y_: train_y})
        test_acc = accuracy.eval(feed_dict={x_: test_x, y_: test_y})
        if (epoch+1) % 10 == 0:
            ed = time.time()
            print("epoch:\t{}\ttotal loss:\t{}\taccuracy:\t{}\tvaridation accuracy:\t{}\ttime:\t{}".format(epoch+1, total_loss, acc, test_acc, ed-st))
            st = time.time()