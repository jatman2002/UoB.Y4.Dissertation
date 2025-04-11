import argparse

from ModelClasses.DQN import DQN
from ModelClasses.PPO import PPO
from ModelClasses.MLP12 import *
from ModelClasses.MLP34 import *
import Results

parser = argparse.ArgumentParser()
parser.add_argument('-r', dest='restaurant', type=int, help='specify which restaurant')
# parser.add_argument('-g', dest='gpu', type=int, default=0, help='specify which gpu to use')
parser.add_argument('-a', dest='algo', type=str, help='specify which algorithm to use')
parser.add_argument('-v', dest='val', action='store_true', default=False, help='use validation set')
parser.add_argument('-R', dest='results', action='store_true', default=False, help='use validation set')
parser.add_argument('-s', dest='start', type=int, default=0, help='start point for MLP')
parser.add_argument('-e', dest='end', type=int, default=1, help='end point for MLP')
args = parser.parse_args()

assert args.restaurant != None, 'Need to specify restaurant with -r'
assert args.algo != None, 'Need to specify algorithm with -a'

algos = {
    'DQN': DQN,
    'PPO': PPO,
    'MLP1': MLP1,
    'MLP2': MLP2,
    'MLP3': MLP3,
    'MLP4': MLP4,
    'MLP': MLP
    }

if args.algo == 'MLP':
    for m in range(args.start, args.end):
        algos[args.algo](args.restaurant, args.val, m).run()
        Results.run(args.restaurant, f'{args.algo}/m')
else:
    algos[args.algo](args.restaurant, args.val).run()

print()

# if args.results:
#     Results.run(args.restaurant, args.algo)