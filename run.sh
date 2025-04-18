#! /bin/bash

# CUDA_VISIBLE_DEVICES=3 python3 src/train.py -r 1 -a MLP1
# CUDA_VISIBLE_DEVICES=3 python3 src/train.py -r 2 -a MLP2
# CUDA_VISIBLE_DEVICES=3 python3 src/train.py -r 3 -a MLP3
# CUDA_VISIBLE_DEVICES=3 python3 src/train.py -r 4 -a MLP4

# CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 1 -a MLP1 -R -v
# CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 1 -a MLP2 -R -v
# CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 1 -a MLP3 -R -v
# CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 1 -a MLP4 -R -v

# CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 1 -a MLP -R
# CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 2 -a MLP -R
# CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 3 -a MLP -R
# CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 4 -a MLP -R
# CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 5 -a MLP -R

# CUDA_VISIBLE_DEVICES=-1 python3 src/train.py -r 1 -a RF
# CUDA_VISIBLE_DEVICES=-1 python3 src/train.py -r 2 -a RF
# CUDA_VISIBLE_DEVICES=-1 python3 src/train.py -r 3 -a RF
# CUDA_VISIBLE_DEVICES=-1 python3 src/train.py -r 4 -a RF
# CUDA_VISIBLE_DEVICES=-1 python3 src/train.py -r 5 -a RF

# CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 1 -a RF -R
# CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 2 -a RF -R
# CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 3 -a RF -R
# CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 4 -a RF -R
# CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 5 -a RF -R

# CUDA_VISIBLE_DEVICES=-1 python3 src/train.py -r 1 -a LR
# CUDA_VISIBLE_DEVICES=-1 python3 src/train.py -r 2 -a LR
# CUDA_VISIBLE_DEVICES=-1 python3 src/train.py -r 3 -a LR
# CUDA_VISIBLE_DEVICES=-1 python3 src/train.py -r 4 -a LR
# CUDA_VISIBLE_DEVICES=-1 python3 src/train.py -r 5 -a LR

CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 1 -a LR -R
CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 2 -a LR -R
CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 3 -a LR -R
CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 4 -a LR -R
CUDA_VISIBLE_DEVICES=-1 python3 src/ModelTesting/run.py -r 5 -a LR -R