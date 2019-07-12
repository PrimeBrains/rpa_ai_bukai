# 2019/07/12 部会


- [パーセプトロンによるXOR実装の可視化を入り口にして、ニューラルネットワークの基礎を理解する](https://qiita.com/masatomix/items/42b322a8db61e5b4d65f)


## まずは、パーセプトロンによる分類を理解

(1,0) → 1 という 2次元から1次元を出力する機会を考える


## つぎに 多層パーセプトロンによる分類を理解

(1,0) → (1,1) → 1 という、2次元 から2次元 そして1次元


実際に、``perceptron_reg.py`` を実行してみる(準備は、動かしてみる、を参照)

##  行列を用いて表現する

一般化

- アフィン変換 a = x⋅W+b
- 活性化関数 z = h(a)

と書ける話。

## 例
### 例1 手書き文字認識

[MNISTの手書き数字の画像セットを、視覚的に見てみる](https://qiita.com/masatomix/items/1ab6aca13b2da96a49fe)

コレは、784次元 → 50 → 100 → 10次元 という処理を行い、手書き文字を 0 〜 9 に分類している例

→ コレって手書き文字認識っていうこと

### 例2 非線形回帰
1次元(x) → 3 → 1次元(y) という処理を行い xからyを出力している

→ 非線形回帰してる


## ネットワークの学習について

記事では 0.5,0.5などの正しく分類ができる値が突然出てきたが、それをどのように導出するかについて。

→ 教師データという、正解の値を持ったデータを用意し、(学習前の)「ニューラルネットが出した値」と「正解の値」の誤差を計算する。つぎにニューラルネットを少し変更してまた誤差を計算する、を繰り返して、最終的に誤差を最小にするニューラルネットをさがす。
(誤差が最小となるニューラルネットが「よくあたるAI」ということ)

→ そのとき「よりニューラルネットのココ(行列の値)を変えれば、誤差が小さくなるよね」という場所を探して変更する
→ その変更ぶんを導出するのに勾配降下法を用いる
参考:[機械学習にでてくる勾配降下法/勾配ベクトルなどの整理。](https://qiita.com/masatomix/items/d4e5fb3b52fa4c92366f)

→ 勾配降下していくのに勾配ベクトルを求める必要があるが、数値微分で計算するとめっちゃ遅いので、別の手法をもちいる (誤差逆伝播法)

それらはまた別の機会に。。。

## 動かしてみる

準備

```
$ python --version
$ python -m venv ./venv
$ source ./venv/bin/activate
(venv)$ pip install numpy matplotlib PyQt5  pillow
```

MNISTの手書き(分類)

```
(venv) $ git clone https://github.com/oreilly-japan/deep-learning-from-scratch.git
(venv) $ cd deep-learning-from-scratch/ch03/
(venv) $ python mnist_show.py

(venv) $ python temp.py
```


回帰ほか

```
適当に移動して
(venv)$ git clone https://github.com/masatomix/machine-learning
(venv)$ cd machine-learning/
(venv)$ checkout feature/init


(venv)$ curl https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch/master/common/functions.py -O

(venv)$ curl https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch/master/common/gradient.py -O
$
```



線形回帰
```
(venv)$ python liner_reg.py
```

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/73777/cbca579b-277c-39c4-c605-cf61867b98d9.png)


非線形回帰
```
(venv)$  python non_liner_reg.py
```

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/73777/b3d1cf69-ef9e-3ec4-b5aa-7afec9ee860f.png)



線形の重回帰

```
(venv) $ python liner2d_reg.py
```

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/73777/0ea69cc4-9aea-06cf-88c3-0bc983624d92.png)



非線形の重回帰

```
(venv)$ python non_liner2d_reg.py
```

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/73777/f44aab0a-d1eb-bc83-f146-7e46d079d6bf.png)
