import torch.nn as nn
import torch.nn.functional as F

from .PytorchModel import PytorchModel


class PPO(PytorchModel):
    def __init__(self, restaurant_name, isVal):
        super().__init__(restaurant_name, 'PPO', PolicyNetwork)

    def run(self):
        super().run()

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return F.softmax(self.l3(x), dim=0)