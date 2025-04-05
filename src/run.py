from RL.DQN import DQN
from RL.PPO import PPO

import ML_methods.MLP1 as MLP1

PPO(1).run()
DQN(1).run()
MLP1.run(1)