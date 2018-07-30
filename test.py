# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 13:11:05 2017

@author: Yuto
"""

import numpy as np
arr = np.array([1,5,10,4,11,22,21,11,10,1])
print (arr)
print()

mat = arr.reshape([5,2])
print (mat)
print()

ze = np.zeros([3,2])
print (ze)
print ()

arr = np.array([0]*10)
print (arr)
print()

x = np.random.randn(10,2)
print (x)
print ()

x = np.random.randn(50000,2)
a = np.average(x)
s = np.std(x)
print (a)
print (s)
print()

print ("aaaaaa")

# 2x2行列a
a = np.array([[2.0, 1.0], [4.0, 2.0]])
# 2x2行列b
b = np.array([[1.0, 1.0], [6.0, 3.0]])
# 行列の和
print (a + b)
print()
# 行列の差
print (a - b)
print()
# 行列の積
print (np.dot(a, b))
print()
# 以下も同様
print (a.dot(b))
print()
# 行列の要素同士の積
print (a * b)
print()
# 行列の要素同士の商
print (a / b)
print()

print("[ 1.  2.  3.]")
x = np.arange(1.0, 4.0)
print (x)

print("# ベクトルのノルム(距離)")
a = np.linalg.norm(x)
print (a)

print("# ベクトルを対角行列にする")
diag_x = np.diag(x)
print(diag_x)

print("# 行列式")
l=np.linalg.det(diag_x)
print (l)

print("# 逆行列")
l=np.linalg.inv(diag_x)
print (l)

print("# 行列の内積")
print("# 3x3単位行列")
e = np.eye(3)
l=np.dot(diag_x, e)
print (l)

print("# 対角和")
e=np.trace(diag_x)
print (e)

print("# 固有値、固有ベクトル")
l=np.linalg.eig(diag_x)
print (l)

# 連立方程式の解
# 2x+y+z = 15
# 4x+6y+3z = 41
# 8x+8y+9z = 83
# 解 : x=5,y=2,z=3
# 薩摩順吉, 四ツ谷晶二, "キーポイント線形代数" p.2より
print("# 連立方程式の解")
a = np.array([[2,1,1],[4,6,3],[8,8,9]])
b = np.array([[15],[41],[83]])
l=np.linalg.solve(a, b)
print (l)





print("# 範囲0〜100の乱数を持つ1次元配列")
x = np.random.randint(0, 100, 10)
print(x)
print("# 範囲0〜100の乱数を持つ4x4行列")
y = np.random.randint(0, 100, (4,4))
print(y)

print (x)
print("# 昇順に整列する")
print("# 注意: np.sort()は破壊的操作")
x.sort()
print (x)




print (y)
print("# yをコピーする")
y1 = np.array(y)
y2 = np.array(y)
print("# 列単位で昇順に整列する")
y1.sort(0)
print (y1)
print("# 行単位で昇順に整列する")
print("# 引数がない場合は、デフォルト")
y2.sort(1)
print (y2)
print()
y.sort()
print(y)





