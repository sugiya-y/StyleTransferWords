#!/bin/bash

for i in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80
do
python test_pre_yahoo_gan.py -m $i -u 0
python test_yahoo.py --output_dir out_ours/yahoo100m_gan_epoch$i
python test_yahoo.py --output_dir "out_ours/yahoo100m_gan_epoch"$i"_colorpre" --color_preserve
done
