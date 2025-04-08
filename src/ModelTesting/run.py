import argparse

from models.DQN import DQN
from models.PPO import PPO
from models.MLP12 import *
from models.MLP34 import *

parser = argparse.ArgumentParser()
parser.add_argument('-r', dest='restaurant', type=int, help='specify which restaurant')
# parser.add_argument('-g', dest='gpu', type=int, default=0, help='specify which gpu to use')
parser.add_argument('-a', dest='algo', type=str, help='specify which algorithm to use')
parser.add_argument('-v', dest='val', action='store_false', default=True, help='use validation set')
args = parser.parse_args()

assert args.restaurant != None, 'Need to specify restaurant with -r'
assert args.algo != None, 'Need to specify algorithm with -a'

algos = {
    'DQN': DQN,
    'PPO': PPO,
    'MLP1': MLP1,
    'MLP2': MLP2,
    'MLP3': MLP3,
    'MLP4': MLP4
    }

algos[args.algo](args.restaurant, args.val).run()