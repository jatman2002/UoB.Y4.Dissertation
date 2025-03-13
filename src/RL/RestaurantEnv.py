import torch

class RestaurantEnv:
    def __init__(self, tables):
        self.tables = tables
        self.state = torch.zeros((len(tables), 64), dtype=torch.float32)

        self.incorrect_table_penalty = -1000

        self.reset()

    def reset(self):
        self.state = torch.zeros((len(self.tables), 64), dtype=torch.int)

    def step(self, action_prob, reservation):

        order_of_tables = torch.argsort(action_prob, descending=True)

        t = 1

        for action_tensor in order_of_tables:
        
            action = action_tensor.item()

            #heavily penalise incorrect tables
            if self.tables.iloc[action]['MinCovers'] > reservation['GuestCount']:
                t+=1
                continue
            if self.tables.iloc[action]['MaxCovers'] < reservation['GuestCount']:
                t+=1
                continue
            if torch.any(self.state[action][int(reservation['BookingStartTime']):int(reservation['EndTime'])] != 0).item():
                t+=1
                continue
            
            self.state[action, int(reservation['BookingStartTime']):int(reservation['EndTime'])] = reservation['BookingCode']

            return action, (100 - self.get_wasted_slots()) / t
        
        return -1, self.incorrect_table_penalty

    def get_wasted_slots(self):
        min_booking_length = 6
        total_wasted_slots = 0
        for table in self.state:
            wasted_slots = 0
            for slot in table:
                if slot.item() == 0:
                    wasted_slots += 1
                else:
                    total_wasted_slots += wasted_slots % min_booking_length
                    wasted_slots = 0

        return total_wasted_slots