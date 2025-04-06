import sys
import os
import logging

from RL.DQN import DQN
from RL.PPO import PPO

original_stdout = sys.stdout

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

print('starting!', file=original_stdout)

sys.stdout = LoggerWriter()

gpu = 5
for r in range(1,6):
    print(f'Restaurant {r}\tPPO', file=original_stdout)
    PPO(r, gpu).run()
    print(f'Restaurant {r}\tDQN', file=original_stdout)
    DQN(r, gpu).run()
