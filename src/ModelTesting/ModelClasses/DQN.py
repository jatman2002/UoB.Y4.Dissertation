import torch.nn as nn
import torch.nn.functional as F

from .PytorchModel import PytorchModel


class DQN(PytorchModel):
    def __init__(self, restaurant_name):
        super().__init__(restaurant_name, 'DQN', DqnNetwork)

    def run(self):
        super().run()

class DqnNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DqnNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, output_size)
        # self.l4 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)