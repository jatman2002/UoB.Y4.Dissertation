import sys
import os
import logging
import argparse

from RL.DQN import DQN
from RL.PPO import PPO
from ML_methods import LRGridSearch, RFGridSearch, MLP1, MLP2, MLP3, MLP4, MLP, LR, RF

parser = argparse.ArgumentParser()
parser.add_argument('-r', dest='restaurant', type=int, help='specify which restaurant')
parser.add_argument('-g', dest='gpu', type=int, default=0, help='specify which gpu to use')
parser.add_argument('-a', dest='algo', type=str, help='specify which algorithm to use')
parser.add_argument('-l', dest='log', action='store_false', default=True, help='disable logging to log file (using the flag prints all to terminal)')
parser.add_argument('-s', dest='start_point',  type=int, default=0, help='start point for grid search for MLP')
args = parser.parse_args()

assert args.restaurant != None, 'Need to specify restaurant with -r'
assert args.algo != None, 'Need to specify algorithm with -a'

algos = {
    'LR': LR.LR,
    'RF': RF.RF,
    'MLP1': MLP1.MLP1,
    'MLP2': MLP2.MLP2,
    'MLP3': MLP3.MLP3,
    'MLP4': MLP4.MLP4,
    'MLP': MLP.MLP,
    'PPO': PPO,
    'DQN': DQN
}

original_stdout = sys.stdout

# Redirect print to logging
class LoggerWriter:
    def write(self, message):
        if message.strip(): 
            logging.info(message.strip())

    def flush(self):
        pass

print('starting!', file=original_stdout)

print(args.log)

if args.log:
    log_file = os.path.join(os.getcwd(), f"myprog-{args.algo}-{args.restaurant}.log")
    open(log_file, "w").close()
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    sys.stdout = LoggerWriter()

algos[args.algo](args.restaurant, args.gpu).run()