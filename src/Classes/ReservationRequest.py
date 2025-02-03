import numpy as np

class ReservationRequest:
    def __init__(self, booking_code, booking_date, start_slot, duration, guest_count, created_on):
        self.booking_code = booking_code
        self.booking_date = booking_date
        self.start_slot = start_slot
        self.duration = duration
        self.guest_count = guest_count
        self.created_on = created_on

        self.assigned = False


    