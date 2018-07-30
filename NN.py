# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:30:17 2017

@author: Yuto
"""

import numpy as np

dim_in = 1              # 入力は1次元
dim_out = 1             # 出力は1次元
hidden_count = 1024     # 隠れ層のノードは1024個
learn_rate = 0.003       # 学習率

# 訓練データは x は -1～1、y は 2 * x ** 2 - 1
train_count = 64        # 訓練データ数
train_x = np.arange(-1, 1, 2 / train_count).reshape((train_count, dim_in))
train_y = np.array([2 * x ** 2 - 1 for x in train_x]).reshape((train_count, dim_out))

# 重みパラメータ。-0.5 〜 0.5 でランダムに初期化。この行列の値を学習する。
w1 = np.random.rand(hidden_count, dim_in) - 0.5
w2 = np.random.rand(dim_out, hidden_count) - 0.5
b1 = np.random.rand(hidden_count) - 0.5
b2 = np.random.rand(dim_out) - 0.5

# 活性化関数は ReLU
def activation(x):
    return np.maximum(0, x)

# 活性化関数の微分
def activation_dash(x):
    return (np.sign(x) + 1) / 2

# 順方向。学習結果の利用。
def forward(x):
    return w2 @ activation(w1 @ x + b1) + b2

# 逆方向。学習
def backward(x, diff):
    global w1, w2, b1, b2
    v1 = (diff @ w2) * activation_dash(w1 @ x + b1)
    v2 = activation(w1 @ x + b1)

    w1 -= learn_rate * np.outer(v1, x)  # outerは直積
    b1 -= learn_rate * v1
    w2 -= learn_rate * np.outer(diff, v2)
    b2 -= learn_rate * diff

# メイン処理
idxes = np.arange(train_count)          # idxes は 0～63
for epoc in range(100000):                # 1000エポック
    np.random.shuffle(idxes)            # 確率的勾配降下法のため、エポックごとにランダムにシャッフルする
    error = 0                           # 二乗和誤差
    for idx in idxes:
        y = forward(train_x[idx])       # 順方向で x から y を計算する
        diff = y - train_y[idx]         # 訓練データとの誤差
        error += diff ** 2              # 二乗和誤差に蓄積
        backward(train_x[idx], diff)    # 誤差を学習
    print(error.sum())                  # エポックごとに二乗和誤差を出力。徐々に減衰して0に近づく。