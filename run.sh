#! /bin/bash

CUDA_VISIBLE_DEVICES=5 python3 src/train.py -r 1 -a MLP
CUDA_VISIBLE_DEVICES=5 python3 src/train.py -r 2 -a MLP
CUDA_VISIBLE_DEVICES=5 python3 src/train.py -r 3 -a MLP
CUDA_VISIBLE_DEVICES=5 python3 src/train.py -r 4 -a MLP
CUDA_VISIBLE_DEVICES=5 python3 src/train.py -r 5 -a MLP

CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 1 -a MLP -v -R
CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 2 -a MLP -v -R
CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 3 -a MLP -v -R
CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 4 -a MLP -v -R
CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 5 -a MLP -v -R