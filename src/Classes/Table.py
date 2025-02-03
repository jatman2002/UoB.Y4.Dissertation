import numpy as np
from Classes import ReservationRequest

class Table:
    def __init__(self, table_code, min_covers, max_covers, slot_count):
        self.table_code = table_code
        self.min_covers = min_covers
        self.max_covers = max_covers
        self.bookings = np.zeros(slot_count)


    def assign_reservation(self, reservation_request):
        for i in range(reservation_request.duration):
            self.bookings[reservation_request.start_slot + i] = reservation_request.booking_code
        reservation_request.assigned = True

    def can_fit_reservation(self, reservation_request):
        can_fit = True
        for i in range(reservation_request.duration):
            can_fit = can_fit and self.bookings[reservation_request.start_slot + i] == 0

        return can_fit