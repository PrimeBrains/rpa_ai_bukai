# 2019/07/12 部会


## 参考リンク

- [パーセプトロンによるXOR実装の可視化を入り口にして、ニューラルネットワークの基礎を理解する](https://qiita.com/masatomix/items/42b322a8db61e5b4d65f)


### まずは、パーセプトロンによる分類を理解

(1,0) → 1 という 2次元から1次元を出力する機会を考える


### つぎに 多層パーセプトロンによる分類を理解

(1,0) → (1,1) → 1 という、2次元 から2次元 そして1次元

###  行列を用いて表現する

一般化


### 例1 手書き文字認識

[MNISTの手書き数字の画像セットを、視覚的に見てみる](https://qiita.com/masatomix/items/1ab6aca13b2da96a49fe)

コレは、784次元 → 50 → 100 → 10次元 という処理を行い、手書き文字を 0 〜 9 に分類している例

→ コレって手書き文字認識っていうこと

### 例2 非線形回帰
1次元(x) → 3 → 1次元(y) という処理を行い xからyを出力している

→ 非線形回帰してる


## ネットワークの学習について

、、、どうする？
