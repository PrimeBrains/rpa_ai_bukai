# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
from PIL import Image

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

x, t = get_data()
network = init_network()

# print(len(x[0]))
y = predict(network, x[0])
print(y)
print(t[0])
p= np.argmax(y) # 最も確率の高い要素のインデックスを取得
print(p)

# def img_show(img):
#     pil_img = Image.fromarray(np.uint8(img))
#     pil_img.show()

# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# img = x_test[0]
# label = t_test[0]
# print(label)  # 7

# print(img.shape)  # (784,)
# img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
# print(img.shape)  # (28, 28)

# img_show(img)