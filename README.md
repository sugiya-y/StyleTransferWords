# privateWork
for private work

このディレクトリの中に色々と作っていきます。
公開しても良いようになったらそのディレクトリだけ別でレポジトリ作って公開しようと思います。

.gitがあるのは

```
/host/space/sugiya-y/research/privateWork
```

## Requirement
- python2.7 (Anaconda install)
- magenta-gpu
- chainer 1.24.0

## arbitrary_image_stylization
Googleのmagentaを利用したスタイル変換を改変して出来たスタイル変換です。
実行をする際には
```
sh ****.sh
```
で実行するようにしています。

### train
まずVGG.modelをtmpに置いておいたほうが早いのでそうします
```
cp ./VGG.model /tmp/
```
学習をします
```
python train_tinynet.py -d *** -u 1
```
-d で出力のディレクトリ名を指定します。指定したディレクトリ名でmodels/の下に出力されます。
-u でVGGを使用するかどうかを指定します。1で使用し、0で使用しません。
