#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Needs argment of 'output_dir'" 1>&2
  exit 1
fi

python test_pre_yahoo_gan.py -m 35 -u 0
python test_yahoo.py "--output_dir out_ours/"$1
python test_yahoo.py "--output_dir out_ours/"$1"_colorpre" --color_preserve

