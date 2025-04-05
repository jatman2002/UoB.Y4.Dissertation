import sys
import os
import logging

from RL.DQN import DQN
from RL.PPO import PPO

log_file = os.path.join(os.getcwd(), "myprog.log")
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

print('starting!')

sys.stdout = LoggerWriter()

PPO(1).run()
DQN(1).run()

PPO(2).run()
DQN(2).run()