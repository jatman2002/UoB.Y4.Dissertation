import numpy as np
from Classes import ReservationRequest

class Table:
    def __init__(self, table_code, min_covers, max_covers, slot_count):
        self.table_code = table_code
        self.min_covers = min_covers
        self.max_covers = max_covers
        self.reservations = [None] * slot_count


    def deepcopy(self):
        t =  Table(self.table_code, self.min_covers, self.max_covers, len(self.reservations))
        t.reservations = self.reservations.copy()
        return t

    def assign_reservation(self, reservation_request):
        for i in range(reservation_request.duration):
            self.reservations[reservation_request.start_slot + i] = reservation_request
        reservation_request.assigned = True

    def can_fit_reservation(self, reservation_request):
        can_fit = True
        for i in range(reservation_request.duration):
            can_fit = can_fit and self.reservations[reservation_request.start_slot + i] == None

        return can_fit

    def get_wasted_table_count(self, minimum_booking_length):
        wasted_table_count = 0
        empty_count = 0
        for slot in self.booking:
            if slot != 0:
                if empty_count < minimum_booking_length and empty_count > 0:
                    wasted_table_count += 1
                empty_count = 0
                continue
            empty_count += 1
        return wasted_table_count
            