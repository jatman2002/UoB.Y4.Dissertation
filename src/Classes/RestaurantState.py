import numpy as np
from Classes import Table

''' The Restaurant class represents the state of a restaurant on a given day. '''

class RestaurantState:
    def __init__(self, site_code, opening_time, closing_time, slot_count, slot_length, tables, reservation_requests, minimum_booking_length):
        self.site_code = site_code
        self.opening_time = opening_time
        self.closing_time = closing_time 

        self.tables = tables
        self.slot_count = slot_count
        self.slot_length = slot_length

        self.reservation_requests = reservation_requests

        self.minimum_booking_length = minimum_booking_length

    def get_wasted_table_count(self):
        wasted_table_count = 0
        for table in self.tables:
            wasted_table_count += table.get_wasted_table_count(minimum_booking_length)
        return wasted_table_count