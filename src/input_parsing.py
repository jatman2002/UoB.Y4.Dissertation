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
        next(reader, None)
        next(reader, None)
        next(reader, None)
        next(reader, None)
        #--------------------------------#

        requests = []

        # change this to datetime.min
        current_date = datetime.strptime('2017-10-02', '%Y-%m-%d').date()

        for row in reader:

            booking_code = int(row[0])
            guest_count = int(row[1])

            booking_date = datetime.strptime(row[2], '%Y-%m-%d').date()

            # change continue to adding to a 2d array maybe?
            # or do what i did before and create a per day list wrapper?
            if booking_date != current_date:
                continue

            start_slot = convert_time_to_slot(int(row[3]) - restaurant.opening_time, restaurant.slot_length)
            duration = convert_time_to_slot(int(row[4]), restaurant.slot_length)
            created_on = datetime.strptime(row[5], '%Y-%m-%d %H:%M:%S.%f')

            request = ReservationRequest(booking_code, guest_count, booking_date, start_slot, duration, created_on)

            requests.append(request)


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



def output_schedule(restaurant):
    with open('C:/git/UoB.Y4.Dissertation/src/output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        output = []
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