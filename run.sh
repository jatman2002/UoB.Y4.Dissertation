#! /bin/bash

python3 src/train.py -r 1 -a MLP1
python3 src/train.py -r 1 -a MLP2
python3 src/train.py -r 1 -a MLP3
python3 src/train.py -r 1 -a MLP4

python3 src/train.py -r 2 -a MLP1
python3 src/train.py -r 2 -a MLP2
python3 src/train.py -r 2 -a MLP3
python3 src/train.py -r 2 -a MLP4

python3 src/train.py -r 3 -a MLP1
python3 src/train.py -r 3 -a MLP2
python3 src/train.py -r 3 -a MLP3
python3 src/train.py -r 3 -a MLP4

python3 src/train.py -r 4 -a MLP1
python3 src/train.py -r 4 -a MLP2
python3 src/train.py -r 4 -a MLP3
python3 src/train.py -r 4 -a MLP4

python3 src/train.py -r 5 -a MLP1
python3 src/train.py -r 5 -a MLP2
python3 src/train.py -r 5 -a MLP3
python3 src/train.py -r 5 -a MLP4