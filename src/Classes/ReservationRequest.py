import numpy as np

class ReservationRequest:
    def __init__(self, booking_code, booking_date, start_slot, duration, guest_count, created_on, assigned=False):
        self.booking_code = booking_code
        self.booking_date = booking_date
        self.start_slot = start_slot
        self.duration = duration
        self.guest_count = guest_count
        self.created_on = created_on

        self.assigned = assigned


    def deepcopy(self):
        return ReservationRequest(self.booking_code, self.booking_date, self.start_slot, self.duration, self.guest_count, self.created_on, self.assigned)