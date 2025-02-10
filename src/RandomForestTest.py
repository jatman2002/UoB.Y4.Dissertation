import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import csv

from dataset_setup import *

from helper import *

def write_schedule(diary, tables, day):
    with open('C:/git/UoB.Y4.Dissertation/src/outputs/RF/' + day.strftime('%Y-%m-%d') + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        output = []
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

reservations, tables = import_dataset()
reservations = feature_engineering(reservations)
train_data, test_data = split_dataset(reservations, tables)

X_train, X_test, y_train, y_test, label_encoder = get_x_y_sets(tables, train_data, test_data)

classifier = RandomForestClassifier(n_estimators=50, max_depth=10)
classifier.fit(X_train, y_train)


# List of unique dates
unique_dates = test_data["BookingDate"].dt.date.unique()
for date in unique_dates:
    reservations_per_day = X_test.loc[test_data['BookingDate'] == date.strftime('%Y-%m-%d')]
    for i, reservation in reservations_per_day.iterrows():
        classifier.predict(reservation)


#---------------------------------------

# y_pred = classifier.predict(X_test)
# y_pred = label_encoder.inverse_transform(y_pred)
# print(y_pred)

# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

#-----------------------------

# output_data = test_data
# output_data['TableCode'] = y_pred


# diary = []
# for table in range(len(tables)):
#     diary.append([None] * 64)

# current_date = None

# for index, booking in output_data.iterrows():

#     if current_date != booking['BookingDate']:

#         if current_date != None:
#             # write csv
#             write_schedule(diary, tables['TableCode'].tolist(), current_date)

#         current_date = booking['BookingDate']

#         diary = []
#         for table in range(len(tables)):
#             diary.append([None] * 64)
    

#     table_index = tables.index[tables['TableCode'] == booking['TableCode']].tolist()[0]
#     start_slot = convert_time_to_slot(booking['BookingTime'] - 36000, 15)
#     duration_in_slots = convert_time_to_slot(booking['Duration'], 15)

#     for i in range(duration_in_slots):
#         diary[table_index][start_slot + i] = str(booking['BookingCode']) + ' - ' + str(duration_in_slots)