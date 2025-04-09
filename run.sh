#! /bin/bash

export CUDA_VISIBLE_DEVICES=-1

python3 src/ModelTesting/run.py -r 1 -a MLP1 -v -R
python3 src/ModelTesting/run.py -r 1 -a MLP2 -v -R
python3 src/ModelTesting/run.py -r 1 -a MLP3 -v -R
python3 src/ModelTesting/run.py -r 1 -a MLP4 -v -R