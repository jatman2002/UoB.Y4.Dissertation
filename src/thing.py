from Classes.RestaurantState import *
from Classes.Table import *
from Classes.ReservationRequest import *

# from Algorithms import Reallocation

import datetime
import numpy as np

# Reallocation.thing()

# print(np.zeros(60))


def datetime_to_slot(dt, opening_time, slot_length):
    return ((dt - opening_time).seconds // (60 * slot_length))

#booking_code, booking_date, start_slot, duration, guest_count, created_on
reservations = [ReservationRequest(1, datetime_to_slot(datetime.fromisoformat(), datetime.fromisoformat(), 15), datetime_to_slot(datetime.fromisoformat(), datetime.fromisoformat(), 15), 4, 3, '03/02/2025')]



# #table_code, min_covers, max_covers, bookings
tables = [Table(i, 2, 4, np.zeros((2300-1400)//15)) for i in range(1,4)]

# #site_code, opening_time, closing_time, slot_count, slot_length, tables, reservation_requests, minimum_booking_length
restaurant_state = RestaurantState(1001, 1400, 2300, (2300-1400)//15, 15, tables, reservations, 4)



# print(f'{restaurant_state.site_code=}')

# new_state = restaurant_state.deepcopy()
# print(f'{new_state.site_code=}')

# new_state.site_code = 1002
# print(f'{new_state.site_code=}')
# print(f'{restaurant_state.site_code=}')
# print(reservations[0].assigned)

# new_res = reservations.copy()
# new_res[0].assigned = True
# new_res.pop()

# print(reservations[0])
# print(new_res[0])