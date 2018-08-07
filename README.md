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
学習をします。すぐ終わります。
```
python train_tinynet.py -d *** -u 1
```
-d で出力のディレクトリ名を指定します。指定したディレクトリ名でmodels/の下に出力されます。
-u でVGGを使用するかどうかを指定します。1で使用し、0で使用しません。

### test
arbitrary_style_transfer/images/valid
の中にある画像ジャイル全てに対して実行を行います。

まず、chainerでパラメータを生成します。
```
python test_pre.py -m *** -u 1
```
-m で学習したモデルの入ったディレクトリ名(train時に指定した -d オプションと同じ名前)を指定します。
-u でVGGを使用するかどうかを指定します。1で使用し、0で使用しません。

次に、生成したパラメータから画像を生成します。
```
python test.py --output_dir out_ours/***　--color_preserve
```
--output_dir で出力するディレクトリを指定します。
--color_preserve で元画像の色を保持します。デフォルトはFalseでこのオプションをつけるとTrueになります。

全体として -u オプションを1にするのは推奨していません。遅い。あんまり結果も変わらない。
GANバージョンでは学習もしてないので0で実行してください。

GAN使ってるバージョンでは。

```
python test_pre_yahoo_gan.py -m *** -u 0
python test_yahoo.py --output_dir out_ours/***　--color_preserve
```
をtest_pre.py, test.pyの代わりにやってください。

### めんどい向け
上のtestの作業は一番うまく行ったと思っているGANmodelのepoch35のmodelで実行するなら
```
./testgan_35.sh ***
```
で実行できます. *** には出力先ディレクトリ名を入れてください。
保存先は out_ours/*** になります。

### その他注意
- ファイル名には画像名、変換に使用した形容詞が含まれています。
- test_pre.py, test_pre_yahoo_gan.pyではクソコードかましてるのでvalidに入れる画像ファイルには0から始まる連番でファイル名をつけてください。
- さらにfor文のiterationも数字を入れなきゃいけないのでvalidの中にあるファイルの数を67行目のrangeに入れてください。ファイル名の最後+1になります。
