#! /bin/bash

CUDA_VISIBLE_DEVICES=2 python3 src/train.py -r 1 -a MLP4
CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 1 -a MLP4 -v -R