import sys
import os
import logging
import argparse

from RL.DQN import DQN

parser = argparse.ArgumentParser()
parser.add_argument('-r', dest='restaurant', type=int)
parser.add_argument('-g', dest='gpu', type=int, default=0)
args = parser.parse_args()

assert args.restaurant != None, 'Need to specify restaurant with -r'

original_stdout = sys.stdout

log_file = os.path.join(os.getcwd(), f"myprog{args.restaurant}.log")
open(log_file, "w").close()
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Redirect print to logging
class LoggerWriter:
    def write(self, message):
        if message.strip(): 
            logging.info(message.strip())

    def flush(self):
        pass

print('starting!', file=original_stdout)

sys.stdout = LoggerWriter()

DQN(args.restaurant, args.gpu).run()
