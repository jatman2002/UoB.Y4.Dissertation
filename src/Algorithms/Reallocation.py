import numpy as np
from Classes.RestaurantState import *
from Classes.ReservationRequest import *
from Classes.Table import *

def run(restaurant_state):
    rejected = 0
    for request in restaurant_state.reservation_requests:
        allocated_reservations = get_allocated_reservations(restaurant_state)
        sort_list(allocated_reservations)
        restaurant_state = reallocate(restaurant_state, allocated_reservations)

        if not find_best_table(request, restaurant_state):
            rejected += 0

    return restaurant_state, rejected


def get_allocated_reservations(restaurant_state):
    allocated_reservations = []
    for table in restaurant_state.tables:
        current_reservation = None
        for slot in table.reservations:
            if slot == None:
                continue
            if current_reservation == None or current_reservation != slot:
                current_reservation = slot
                allocated_reservations.append(slot)
    return allocated_reservations


def sort_list(allocated_reservations):
    allocated_reservations.sort(key=lambda r: r.start_slot)
    pass


def clear_bookings(state, to_remove):
    for table in state.tables:
        for slot in range(len(table.reservations)):
            if table.reservations[slot] != None and table.reservations[slot] in to_remove:
                table.reservations[slot] = None


def reallocate(restaurant_state, allocated_reservations):
    state = restaurant_state.deepcopy()
    clear_bookings(state, allocated_reservations)

    for reservation in allocated_reservations:
        if not find_best_table(reservation, state):
            new_allocated_reservations = allocated_reservations.copy()
            new_allocated_reservations.remove(reservation)
            state = reallocate(state, new_allocated_reservations)
            break
    
    return state


def find_best_table(request, state):
    best_table = -1
    leastWastedSlots = float('inf')

    for table in state.tables:
        if not table.can_fit_reservation(request):
            continue
        if not (table.min_covers <= request.guest_count or request.guest_count <= table.max_covers):
            continue
        wasted_slots = get_wasted_slots(state, table, request)
        if wasted_slots < leastWastedSlots:
            best_table = table
            leastWastedSlots = wasted_slots

    if best_table == -1:
        return False

    best_table.assign_reservation(request)
    return True


def get_wasted_slots(state, table, request):
    start_slot = request.start_slot
    end_slot = request.start_slot + request.duration - 1

    wasted_slots_before = 0
    slot = start_slot - 1
    while slot >= start_slot - state.minimum_booking_length and slot >= 0:
        if table.reservations[slot] is not None:
            break
        wasted_slots_before += 1
        slot -= 1

    wasted_slots_before %= state.minimum_booking_length

    wasted_slots_after = 0
    slot = end_slot + 1
    while slot <= end_slot + state.minimum_booking_length and slot < state.slot_count:
        if table.reservations[slot] is not None:
            break
        wasted_slots_after += 1
        slot += 1
    wasted_slots_after %= state.minimum_booking_length

    return wasted_slots_before + wasted_slots_after