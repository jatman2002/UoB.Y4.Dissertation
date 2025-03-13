import torch

class RestaurantEnv:
    def __init__(self, tables):
        self.tables = tables
        self.state = torch.zeros((len(tables), 64), dtype=torch.float32)

        self.incorrect_table_penalty = -1

        self.reset()

    def reset(self):
        self.state = torch.zeros((len(self.tables), 64), dtype=torch.int)

    def step(self, action, reservation):
        
        #heavily penalise incorrect tables
        if self.tables.iloc[action]['MinCovers'] > reservation['GuestCount']:
            return self.incorrect_table_penalty
        if self.tables.iloc[action]['MaxCovers'] < reservation['GuestCount']:
            return self.incorrect_table_penalty
        if torch.any(self.state[action][int(reservation['BookingStartTime']):int(reservation['BookingEndTime'])] != 0):
            return self.incorrect_table_penalty
        
        self.state[action, int(reservation['BookingStartTime']):int(reservation['BookingEndTime'])] = reservation['BookingCode']

        return 100 - self.get_wasted_slots()

    def get_wasted_slots(self):
        min_booking_length = 6
        total_wasted_slots = 0
        for table in self.state:
            wasted_slots = 0
            for slot in table:
                if slot == None:
                    wasted_slots += 1
                else:
                    total_wasted_slots += wasted_slots % min_booking_length
                    wasted_slots = 0

        return total_wasted_slots