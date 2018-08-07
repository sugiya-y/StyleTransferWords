#!/bin/bash

python test_pre_yahoo_gan.py -m 35 -u 0
python test_yahoo.py --output_dir out_ours/yahoo100m_gan_epoch35
python test_yahoo.py --output_dir out_ours/yahoo100m_gan_epoch35_colorpre --color_preserve
