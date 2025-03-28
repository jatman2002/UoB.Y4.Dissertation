import torch

class RestaurantEnv:
    def __init__(self, tables, device):
        self.tables = tables
        self.state = torch.zeros((len(tables), 64), dtype=torch.float32, device=device)

        self.incorrect_table_penalty = -1000

        self.reset(device)

    def reset(self, device):
        self.state = torch.zeros((len(self.tables), 64), dtype=torch.int, device=device)

    def step(self, action_prob, reservation):

        if torch.argmax(action_prob).item() == len(self.tables):
            return len(self.tables), self.incorrect_table_penalty

        order_of_tables = torch.argsort(action_prob[:len(self.tables)], descending=True)

        t = 1

        start = int(reservation['BookingStartTime'])
        end = int(reservation['EndTime'])

        for action_tensor in order_of_tables:
        
            action = action_tensor.item()

            #heavily penalise incorrect tables
            if self.tables.iloc[action]['MinCovers'] > reservation['GuestCount']:
                t+=1
                continue
            if self.tables.iloc[action]['MaxCovers'] < reservation['GuestCount']:
                t+=1
                continue
            if torch.any(self.state[action][start:end] != 0).item():
                t+=1
                continue
            
            self.state[action, start:end] = reservation['BookingCode']

            return action, (100 - self.get_wasted_slots()) / t
        
        return len(self.tables), self.incorrect_table_penalty

    def get_wasted_slots(diary):
        min_booking_length = 6
        total_wasted_slots = 0
        wasted_count = 0
        for table in diary:
            empty_slots = 0
            for slot in table:
                # if slot == 0:
                #     wasted_slots += 1
                # else:
                #     total_wasted_slots += wasted_slots % min_booking_length
                #     wasted_count += 1
                #     wasted_slots = 0

                if slot != 0:
                    if empty_slots < min_booking_length and empty_slots > 0:
                        wasted_count += 1
                    empty_slots = 0
                    continue
                empty_slots += 1

        return wasted_count