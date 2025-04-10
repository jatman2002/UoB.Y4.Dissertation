#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python3 src/train.py -r 1 -a MLP
CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 1 -a MLP -v -R -e 12