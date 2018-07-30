# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 16:03:03 2018

@author: Yuto
"""

import tensorflow as tf

height,width = 32,32
CLASS = 2

X = tf.placeholder(tf.float32,[None, height * width])
Y = tf.placeholder(tf.float32,[None, CLASS])
keep_prob = tf.placeholder(tf.float32)

import glob,random,cv2
import numpy as np
"""
画像はOpenCVで読み込みます。これはshapeが(height, width, channel)となっていてtensorflowのネットワークのshape(minibatch, height, width, channel)と相性がいいからです。
正解ラベルはファイル名のインデックスをとって作ります。この時、正解ラベルは1-hotベクトルとして作ります。
関数の引数は画像があるディレクトリです。今回は'./Images/Train/'です。
関数から返ってくるのは画像と正解ラベルが入ってリストです。こんな構造になります。[画像のnumpy, 正解のnumpy]
"""

def load_images(dataset_path, shuffle=True):
    filepaths_jpg = glob.glob(dataset_path + '/*.jp*g')
    filepaths_png = glob.glob(dataset_path + '/*.png')
    filepaths_bmp = glob.glob(dataset_path + '/*.bmp')
    filepaths = filepaths_jpg + filepaths_png + filepaths_bmp
    filepaths.sort()
    dataset = []
    imgs = np.zeros((len(filepaths), height, width, 3), dtype=np.float32)
    gts = np.zeros((len(filepaths), CLASS), dtype=np.float32)
    
    print(filepaths )
    
    for i,filepath in enumerate(filepaths):
        
        img = cv2.imread(filepath)
        img = cv2.resize(img,(width,height))
        
        img = img / 255.
        
        label = int(filepath.split('\\')[-1].split('_')[0])
        
        imgs[i] = img
        gts[i,label] = 1.
           
    inds = np.arange(len(filepaths))
    
    
    if shuffle:
        random.shuffle(inds)
    
    imgs = imgs[inds]
    gts = gts[inds]
    
    dataset = [imgs, gts]
    
    return dataset

train_data = load_images('Image/Train/')
"""
train = datasets[:1000] # 最初の千個を学習用
test = datasets[1000:1100] # 千個めから1100個目までをテスト用
"""
"""
print(train_data[0][:])
print(train_data[1][:])
print(train_data[1:2])
"""

#Convolution
def conv2d(x,ksize=3,in_num=1, out_num=32, strides=1, bias=True):
    W = tf.Variable(tf.random_normal([ksize,ksize,in_num,out_num]))
    x = tf.nn.conv2d(x,W, strides=[1,strides, strides, 1], padding='SAME')
    
    if bias:
        b = tf.Variable(tf.random_normal([out_num]))
        x = tf.nn.bias_add(x,b)
        
    return tf.nn.relu(x)

#Pooling
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

#Fully Connected
def fc(x, in_num=100, out_num=100, bias=True):
    x = tf.matmul(x,tf.Variable(tf.random_normal[out_num]))
    
    if bias:
        b = tf.Variable(tf.random_normal([out_num]))
        x = tf.add(x,b)
        
    return x

#ネットワーク全体
def conv_net(x):
    x = tf.reshape(x,shape=[-1, height, width, 3])
    
    conv1 = conv2d(x, ksize=5, in_num=1, out_num=32)
    pool1 = maxpool2d(conv1, k=2)
    conv2 = conv2d(pool1, ksize=5, in_num=32, out_num=64)
    pool2 = maxpool2d(conv2, k=2)
    
    mb, h, w, c = pool2.get_shape().as_list()
    feature_shape = h * w * c
    flat = tf.reshape(pool2, [-1,feature_shape])
    
    fc1 = fc(flat, in_num=feature_shape, out_num=1024)
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1,keep_drob=0.5)
    
    out = fc(fc1, in_num=1024, out_num=CLASS)
    
    return out

"""学習の準備"""
"""
ここから学習するまでの準備をしていきます。まずはネットワークを作ります。出力にsoftmaxをかけます。
"""

logits = conv_net(X)#network
prediction = tf.nn.softmax(logits)

#lossの定義
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = Y))

#optimizerの定義
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()


"""ミニバッチを作る関数
ミニバッチを作る関数を作成しなければなりません。こんな感じで作ります。
dataはload_images()で読み込んだデータ、batchはこちらで指定するミニバッチサイズです。lastは前回のミニバッチ
"""

def get_batch(data, batch, last):
    imgs, gts = data
    
    data_num = len(imgs)
    ind = last + batch
    
    if ind < data_num:
        img = imgs[last : ind]
        gt  = gts[last : img]
        last = ind
    else:
        resi = ind = data_num
        img1, gt1 = imgs[last:], gts[last:]
        img2, gt2 = imgs[:resi], gts[:resi]
        img = np.vstack((img1, img2))
        print(gt1.shape, gt2.shape)
        gt = np.vstack((gt1, gt2))
        last = resi

    return img, gt, last

with tf.Session() as sess:
    sess.run(init)
    
    last = 0
    
    for step in range(1000):
        with tf.variable_scope('scope-{}'.format(step)):
            step += 1;
            batch_x, batch_y, last = get_batch(train_data, 32, last)
            
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
    
            if step % 10 == 0 or step == 1:
                loss, acc = sess.run([loss_op, accuracy],
                                     feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
    
                print('Step: {}, Loss: {}, Accuracy: {}'.format(step, loss, acc))
        
        print('Training finished!!')
        
        #print(sess.run(accuracy,feed_dict={X: mnist.test.images[:256], Y: mnist.test.labels[:256],keep_prob: 1.0}))
        