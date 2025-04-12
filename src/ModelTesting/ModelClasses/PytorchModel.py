import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from .Model import Model

class PytorchModel(Model):
    def __init__(self, restaurant_name, name, network):
        self.name = name
        self.restaurant_name = restaurant_name
        self.features = ['GuestCount', 'BookingStartTime', 'Duration', 'EndTime']
        self.test, self.tables = self.load_data(False)
        self.model = None
        self.load_model(network, len(self.features) + len(self.tables)*64, len(self.tables))
        self.file_path = f'Restaurant-{restaurant_name}/{name}'


    def load_model(self, network, inp, out):
        self.model = network(inp,out)
        self.model.load_state_dict(torch.load(f'{os.getcwd()}/models/{self.name}/models/{self.name}-R-{self.restaurant_name}.pt', map_location='cpu'))

    def find_table(self, reservation, diary):
        res_details = torch.tensor(reservation.astype(float).values, dtype=torch.float32)
        state_details = (diary.flatten() != 0).int()

        with torch.no_grad():
            actions = torch.argsort(self.model(torch.cat((res_details, state_details))), descending=True).tolist()

            for a_i in range(len(actions)):
                start = int(reservation['BookingStartTime'])
                end = int(reservation['EndTime'])

                a = actions[a_i]
                #heavily penalise incorrect tables
                if self.tables.iloc[a]['MinCovers'] > reservation['GuestCount']:
                    continue
                if self.tables.iloc[a]['MaxCovers'] < reservation['GuestCount']:
                    continue
                if torch.any(diary[a][start:end] != 0).item():
                    continue
                return a
            return -1

    def reset_diary(self):
        return torch.zeros((len(self.tables), 64), dtype=torch.int)