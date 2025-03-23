import csv
from datetime import datetime

from Classes.RestaurantState import RestaurantState
from Classes.ReservationRequest import ReservationRequest
from Classes.Table import Table

from helper import *


def parse_reservations(restaurant, file): #'C:/git/UoB.Y4.Dissertation/src/Restaurant-1/reservations.csv'

    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        next(reader, None)

        #------testing with one day------#
        # next(reader, None)
        # next(reader, None)
        # next(reader, None)
        # next(reader, None)
        #--------------------------------#

        requests = []
        day_requests = []

        # change this to datetime.min
        current_date = datetime.min

        for row in reader:

            booking_code = int(row[0])
            guest_count = int(row[1])

            booking_date = datetime.strptime(row[2], '%Y-%m-%d').date()

            # if booking_date != datetime.strptime('2020-08-31', '%Y-%m-%d').date():
            #     continue

            if booking_date != current_date:
                current_date = booking_date
                if len(day_requests) > 0:
                    requests.append(day_requests)
                day_requests = []

            start_slot = convert_time_to_slot(int(row[3]) - restaurant.opening_time, restaurant.slot_length)
            duration = convert_time_to_slot(int(row[4]), restaurant.slot_length)
            created_on = datetime.strptime(row[5], '%Y-%m-%d %H:%M:%S.%f')

            request = ReservationRequest(booking_code, guest_count, booking_date, start_slot, duration, created_on)

            day_requests.append(request)

        requests.append(day_requests)

    return requests

def parse_tables(restaurant, file): #'C:/git/UoB.Y4.Dissertation/src/Restaurant-1/tables.csv'
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        next(reader, None)

        tables = []

        for row in reader:
            table_code = int(row[1])
            min_covers = int(row[2])
            max_covers = int(row[3])

            table = Table(table_code, min_covers, max_covers, restaurant.slot_count)

            tables.append(table)

    restaurant.tables = tables



def output_schedule(restaurant, day, rejected):
    with open('C:/git/UoB.Y4.Dissertation/src/outputs/Restaurant-1/Non-ML/' + day.strftime('%Y-%m-%d') + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        output = [['reservations: ', len(restaurant.reservation_requests), 'rejections:', rejected, 'wasted slots', get_wasted_slots(restaurant.tables)],[]]
        for table in restaurant.tables:
            table_output = []
            table_output.append(table.table_code)
            for slot in table.reservations:
                if slot == None:
                    table_output.append('')
                else:
                    table_output.append(slot.booking_code)
            output.append(table_output)

        writer.writerows(output)


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