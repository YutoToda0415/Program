# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 16:00:56 2017

@author: Yuto
"""
from random import choice
import tensorflow as tf
import numpy as np
version = tf.VERSION
print (version)


const_a = tf.constant([[1, 2, 3], [4, 5, 6]], shape=(2,3),verify_shape=True)
print (const_a)
print("ok")

sess = tf.Session()
result_a = sess.run(const_a)
print ("---------")
print (result_a)

varible_a = tf.Variable([[1.0,1.0],[2.0,2.0]])
print (varible_a)

sess = tf.Session()
print("#Initializerで初期化を行う")
sess.run(varible_a.initializer)

result_a = sess.run(varible_a)

print("_______")
print (result_a)

ph_a = tf.placeholder(tf.int16)
ph_b= tf.placeholder(tf.int16)

add_op=tf.add(ph_a,ph_b)

sess = tf.Session()
result_a1= sess.run(add_op,feed_dict={ph_a:2,ph_b:3})
print (result_a1)


x_data = np.random.randn(3,3)
x= tf.constant(x_data, shape=(3,3))

sess = tf.Session()


print ("--run--")
print (sess.run(x))

print("型(dtype)")
print(x.dtype)

print("--shape--次元数")
print (sess.run(tf.shape(x)))

print ("--rank(ランク)")
print (sess.run(tf.rank(x)))

print ("size")
print (sess.run(tf.size(x)))


x = tf.placeholder(tf.float32,shape=(2,2))
y = tf.placeholder(tf.float32,shape=(2,2))

mat_a = np.arange(1,5).reshape(2,2)
mat_b = np.arange(11,15).reshape(2,2)

add_op = tf.add(x,y)

sess=tf.Session()
result_mat = sess.run(add_op, feed_dict = {x:mat_a,y:mat_b})

print (result_mat)

sub_op = tf.subtract(x,y)
result_mat = sess.run(sub_op,feed_dict={x:mat_a,y:mat_b})
print (result_mat)

input_data= np.array([choice([True,False]) for _ in range(16)]).reshape(4,4)
where_op = tf.where(input_data)

ses = tf.Session()

print (input_data)
print (ses.run(where_op))


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("name",None,"なまえ")
tf.app.flags.DEFINE_integer("int_value",5,"整数")

if __name__ == "__main__":
        print (FLAGS.name)
        print (FLAGS.int_value)
        
        # 3x3行列の乱数生成オペレーション
rand_op = tf.random_normal(shape=(3,3))
# 3x3行列のVariable このノードが保存される
x = tf.Variable(tf.zeros(shape=(3,3)))
# xに3x3の乱数行列を割り当てるオペレーション
update_x = tf.assign(x, rand_op)

# セッションの保存・読み込みを行うオブジェクト
saver = tf.train.Saver()

# 保存用のセッション
# rand_opの実行ごとにxノードには違う乱数が格納される
# そのときのセッションが保存される
with tf.Session() as sess1:
  sess1.run(tf.global_variables_initializer())
  for i in range(0, 3):
    # rand_opを実行して、3x3行列を生成し、xに割り当てる
    sess1.run(update_x)
    # xの値を表示する
    print ("--save ./rand--")
    print (sess1.run(x))
    # セッション情報を保存する
    saver.save(sess1, "./rand", global_step=i)

# セッションの読み込み
with tf.Session() as sess2:
  sess2.run(tf.global_variables_initializer())
  # 最後のセッションを読み込む
  saver.restore(sess2, "./rand-2")
  print ("--load ./rand-2--")
  print (sess2.run(x))
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
