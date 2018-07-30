# ここにあるファイルの説明

yahoo100m_txt.pickle: yahoo100M全文textをひとかたまりでpickleにしたやつ.高速ロード用.

yahoo100m_txt_lines.pickle: yahoo100M全文をlistの形でpickleにしたやつ.for回したいとき用.

count_docs_yahoo100M.py: yahoo100M全文から形容詞だけを抽出して出現頻度順でソートします. -> yahoo100m_adv.txt, yahoo100m_adv_num.txt

yahoo100m_adv.txt: yahoo100Mから形容詞だけを抽出したtext.

yahoo100m_adv_num.txt: yahoo100Mから抽出した形容詞と出現数を並べたtext.

dataselector.py: 形容詞リストのtextと全文から画像IDと形容詞の対応付をしたlistを生成する.

reduce_advs.py: word2vecの辞書に含まれている単語のみを抽出する -> yahoo100m_adv_exists.txt, yahoo100m_adv_exists.pickle

yahoo100m_adv_exists.txt: word2vecに含まれてる単語のtext. 可視化用.

yahoo100m_adv_exists.pickle: word2vecに含まれている単語のpickle. listの形で保存されている. ロード用.