import numpy as np

import datetime

from Classes.Table import Table
from Classes.ReservationRequest import ReservationRequest

''' The Restaurant class represents the state of a restaurant on a given day. '''

class RestaurantState:
    def __init__(
        #self, site_code, opening_time, slot_count, slot_length, minimum_booking_length):

        self, opening_time, slot_count, slot_length, minimum_booking_length):

        # self.site_code = site_code
        self.opening_time = opening_time

        self.tables = []
        self.slot_count = slot_count
        self.slot_length = slot_length

        self.reservation_requests = []

        self.minimum_booking_length = minimum_booking_length

    def deepcopy(self):
        r =  RestaurantState(
            self.opening_time,
            self.slot_count, 
            self.slot_length,
            self.minimum_booking_length)

        r.tables = [t.deepcopy() for t in self.tables]
        r.reservation_requests = [r.deepcopy() for r in self.reservation_requests]

        return r

    def get_wasted_table_count(self):
        wasted_table_count = 0
        for table in self.tables:
            wasted_table_count += table.get_wasted_table_count(minimum_booking_length)
        return wasted_table_count