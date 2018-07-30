# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:45:39 2017

@author: Yuto
"""

import tensorflow as tf
import numpy as np

dataset = np.genfromtxt("./bezdekIris.data",delimiter=',', dtype = [float,float,float,float,"S32"])

np.random.shuffle(dataset)

def get_labels(dataset):
    """ラベル（正解データ）を1lfkベクトル"""
    raw_labels = [item[4] for item in dataset]
    
    labels = []
    
    for l in raw_labels:
        if l =="Iris-setosa":
            labels.append([1,0,0])
            
        elif l =="Iris-versicolor":
            labels.append([0,1,0])
        
        elif l == "Iris-virginica":
            labels.append([0,0,1])

    return np.array(labels)

def get_data(dataset):
    raw_data = [list(item)[:4] for item in dataset]
    
    return np.array(raw_data)

labels = get_labels(dataset)

# データ
data = get_data(dataset)

# irisデータセットの形式
print (labels.shape)
print (data.shape)

# 訓練データとテストデータに分割する
# 訓練用データ
train_labels = labels[:120]
train_data = data[:120]
print (train_labels.shape)
print (train_data.shape)

# テスト用データ
test_labels = labels[120:]
test_data = data[120:]
print (test_labels.shape)
print (test_data.shape)

print(test_data)


"""機会学習"""

#t = tf.placeholder(tf.float32,shape=(np.newaxis,3))
#X = tf.placeholder(tf.float32,shape=(np.newaxis,4))

t = tf.placeholder(tf.float32,shape=(120,3))
X = tf.placeholder(tf.float32,shape=(120,4))


def single_layer(x):
    """隠れそう"""
    node_num = 1024
    w = tf.Variable(tf.truncated_normal([4,node_num]))
    b = tf.Variable(tf.zeros([node_num]))
    f = tf.matmul(X,w) + b
    layer = tf.nn.relu(f)
    return layer

def output_layer(layer):
    
    node_num = 1024
    w = tf.Variable(tf.zeros([node_num,3]))
    b = tf.Variable(tf.zeros([3]))
    f = tf.matmul(layer,w) + b
    p = tf.nn.softmax(f)
    return p

# 隠れ層
hidden_layer = single_layer(X)
# 出力層
p = output_layer(hidden_layer)

# 交差エントロピー
cross_entropy = t * tf.log(p)
# 誤差関数
loss = -tf.reduce_mean(cross_entropy)
# トレーニングアルゴリズム
# 勾配降下法 学習率0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_step = optimizer.minimize(loss)
# モデルの予測と正解が一致しているか調べる
correct_pred = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
# モデルの精度
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


### 学習の実行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    i = 0
    for i in range(2000):
        i += 1
        # トレーニング
        sess.run(train_step, feed_dict={X:train_data,t:train_labels})
        # 200ステップごとに精度を出力
        if i % 200 == 0:
            # コストと精度を出力
            train_loss, train_acc = sess.run([loss, accuracy], feed_dict={X:train_data,t:train_labels})
            # テスト用データを使って評価
            test_loss, test_acc = sess.run([loss, accuracy], feed_dict={X:test_data,t:test_labels})
            print ("Step: ",i)
            print ("[Train] cost: ", train_loss," acc: ",train_acc)
            print ("[Test] cost: ",test_loss, "acc: ", test_acc)






