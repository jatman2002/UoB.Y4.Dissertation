import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import csv

def test_predictor(reservations, tables, predictor, get_best_table):

    booking_date_as_dt = pd.to_datetime(reservations['BookingDate']).dt.date
    unique_days = booking_date_as_dt.unique()

    for day in unique_days:
        reservations_for_day = reservations.loc[booking_date_as_dt == day]

        diary = []
        for i in range(len(tables)):
            diary.append([None] * 64)

        for i in range(len(reservations_for_day)):
            reservation = reservations_for_day.iloc[i]
            
            best_table_index = get_best_table(predictor, reservation.drop(['TableCode', 'BookingDate', 'CreatedOn', 'BookingCode']), diary)

            booking_code = str(reservations.loc[reservations.index == reservation.name].iloc[0]['BookingCode'])
            for i in range(int(reservation['Duration'])):
                diary[best_table_index][int(reservation['BookingTime']) + i] = booking_code
            # y_pred.append(tables.iloc[best_table_index]['TableCode'])

        write_schedule(diary, tables['TableCode'].tolist(), day, len(reservations_for_day))


def write_schedule(diary, tables, day, num_reservations):
    with open('C:/git/UoB.Y4.Dissertation/src/outputs/RF2/' + day.strftime('%Y-%m-%d') + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        output = [[num_reservations],[]]
        for table_code, table in zip(tables, diary):
            table_output = []
            table_output.append(table_code)
            for slot in table:
                if slot == None:
                    table_output.append('')
                else:
                    table_output.append(slot)
            output.append(table_output)

        writer.writerows(output)