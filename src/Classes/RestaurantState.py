import numpy as np
from Classes import Table

''' The Restaurant class represents the state of a restaurant on a given day. '''

class RestaurantState:
    def __init__(self, site_code, opening_time, closing_time, slot_count, slot_length, table_count, reservation_requests):
        self.site_code = site_code
        self.opening_time = opening_time
        self.closing_time = closing_time 

        self.tables = np.zeros(table_count)
        self.slot_count = slot_count
        self.slot_length = slot_length

        self.reservation_requests = reservation_requests

    def add_table(self, table_code, min_covers, max_covers):
        table = Table(table_code, min_covers, max_covers, self.slot_count)
        self.tables.append(table)