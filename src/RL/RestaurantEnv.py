import torch

class RestaurantEnv:
    def __init__(self, tables, device):
        self.tables = tables
        self.state = torch.zeros((len(tables), 64), dtype=torch.float32, device=device)

        self.incorrect_table_penalty = -100
        self.wrong_table_size = -50
        self.table_is_full = -50

        self.reset(device)

    def reset(self, device):
        self.state = torch.zeros((len(self.tables), 64), dtype=torch.int, device=device)

    def step(self, action_list, reservation):

        start = int(reservation['BookingStartTime'])
        end = int(reservation['EndTime'])

        reward = 0

        for a in range(len(action_list)):
            action = action_list[a]

            if action == len(self.tables):
                return len(self.tables), self.incorrect_table_penalty

            #heavily penalise incorrect tables
            if self.tables.iloc[action]['MinCovers'] > reservation['GuestCount']:
                reward += self.wrong_table_size
                continue
            if self.tables.iloc[action]['MaxCovers'] < reservation['GuestCount']:
                reward +=  self.wrong_table_size
                continue
            if torch.any(self.state[action][start:end] != 0).item():
                reward +=  self.table_is_full
                continue
            
            self.state[action, start:end] = reservation['BookingCode']
            return action, (200 - self.get_wasted_slots()*0.1 + reward)/(a+1)

        return len(self.tables), self.incorrect_table_penalty

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