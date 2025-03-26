import torch

class RestaurantEnv:
    def __init__(self, tables, device):
        self.tables = tables
        self.state = torch.zeros((len(tables), 64), dtype=torch.float32, device=device)

        self.incorrect_table_penalty = -75
        self.wrong_table_size = -50
        self.table_is_full = -50

        self.reset(device)

    def reset(self, device):
        self.state = torch.zeros((len(self.tables), 64), dtype=torch.int, device=device)

    def step(self, action, reservation):

        if action == len(self.tables):
            return self.incorrect_table_penalty

        start = int(reservation['BookingStartTime'])
        end = int(reservation['EndTime'])

        #heavily penalise incorrect tables
        if self.tables.iloc[action]['MinCovers'] > reservation['GuestCount']:
            return self.wrong_table_size
        if self.tables.iloc[action]['MaxCovers'] < reservation['GuestCount']:
            return self.wrong_table_size
        if torch.any(self.state[action][start:end] != 0).item():
            return self.table_is_full
        
        self.state[action, start:end] = reservation['BookingCode']

        return 100 - self.get_wasted_slots()

    def get_wasted_slots(self):
        min_booking_length = 6
        total_wasted_slots = 0
        wasted_slots = 0
        for table in self.state:
            for slot in table:
                if slot == 0:
                    wasted_slots += 1
                else:
                    total_wasted_slots += wasted_slots % min_booking_length
                    wasted_slots = 0
        return total_wasted_slots