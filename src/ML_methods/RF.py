import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from Test import *

def find_table(predictor, reservation, diary):
    probs = predictor.predict_proba(pd.DataFrame([reservation]))[0]
    # filter out unavailable tables
    for i, t in tables.iterrows():
        # 1. size constraint
        if t['MinCovers'] > reservation['GuestCount'] or t['MaxCovers'] < reservation['GuestCount']:
            probs[i] = 0
            continue

        # 2. time constraint
        if np.all(diary[i][int(reservation['BookingTime']):int(reservation['EndTime'])] != None):
            probs[i] = 0

        if probs.sum() <= 0:
            continue
        probs /= probs.sum()

    best_table_index = np.argmax(probs)
    return best_table_index

reservations = pd.read_csv('C:/git/UoB.Y4.Dissertation/src/Restaurant-1/output.csv')
tables = pd.read_csv('C:/git/UoB.Y4.Dissertation/src/Restaurant-1/tables.csv')

# print(reservations)
# print([t for t in tables])

booking_date = pd.to_datetime(reservations['BookingDate'])

reservations['BookingDateDayOfWeek'] = booking_date.dt.dayofweek
reservations['BookingDateDay'] = booking_date.dt.day
reservations['BookingDateMonth'] = booking_date.dt.month
reservations['BookingDateYear'] = booking_date.dt.year

reservations["IsPeakTime"] = reservations["BookingTime"].apply(lambda x: 1 if 18 <= x <= 21 else 0)


reservations['BookingTime'] = (reservations['BookingTime'] - 36000) / (60*15)
reservations['Duration'] = reservations['Duration'] / (60*15)
reservations["EndTime"] = reservations["BookingTime"] + reservations["Duration"]

# Get unique days and shuffle them
unique_days = booking_date.dt.date.unique()
# np.random.shuffle(unique_days)

# Split 70% for training, 30% for testing
split_idx = int(len(unique_days) * 0.7)
train_days, test_days = unique_days[:split_idx], unique_days[split_idx:]

# Create train and test sets based on BookingDate
train_data = reservations[booking_date.dt.date.isin(train_days)]
test_data = reservations[booking_date.dt.date.isin(test_days)]

# features = ['GuestCount', 'BookingDateDayOfWeek','BookingDateDay', 'BookingDateMonth', 'BookingTime', 'Duration', "IsPeakTime", "EndTime"]
# for t in tables['TableCode']:
#     features.append(f'is{t}Free')
#     features.append(f'{t}MinCovers')
#     features.append(f'{t}MaxCovers')

X_train, y_train = train_data.drop(columns=['TableCode', 'BookingDate', 'CreatedOn', 'BookingCode'], axis=1), train_data["TableCode"]
# X_test, y_test = test_data.drop(columns=['TableCode', 'BookingDate', 'CreatedOn', 'BookingCode'], axis=1), test_data["TableCode"]

# label_encoder = LabelEncoder()
# label_encoder.fit(tables)

# y_train, y_test = label_encoder.transform(y_train), label_encoder.transform(y_test)

classifier = RandomForestClassifier(n_estimators=100, max_depth=20)
classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)
# # y_pred = label_encoder.inverse_transform(y_pred)
# print(y_test)
# print(y_pred)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

# probs = classifier.predict_proba(X_test.iloc[[0]])[0]

y_pred = []

test_predictor(test_data, tables, classifier, find_table)

# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")