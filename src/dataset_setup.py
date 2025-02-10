import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def import_dataset():
    # Load dataset
    reservations = pd.read_csv('C:/git/UoB.Y4.Dissertation/src/Restaurant-1/existing.csv')
    tables = pd.read_csv('C:/git/UoB.Y4.Dissertation/src/Restaurant-1/tables.csv')

    reservations = reservations.merge(tables, on="TableCode", how="left")

    return reservations, tables

def feature_engineering(reservations):
    reservations = reservations.drop('SiteCode', axis=1)

    reservations['BookingDate'] = pd.to_datetime(reservations['BookingDate'])

    reservations['BookingDateDayOfWeek'] = reservations['BookingDate'].dt.dayofweek
    reservations['BookingDateDay'] = reservations['BookingDate'].dt.day
    reservations['BookingDateMonth'] = reservations['BookingDate'].dt.month
    reservations['BookingDateYear'] = reservations['BookingDate'].dt.year

    return reservations

def split_dataset(reservations, tables):

    # Get unique days and shuffle them
    unique_days = reservations["BookingDate"].dt.date.unique()
    np.random.shuffle(unique_days)

    # Split 70% for training, 30% for testing
    split_idx = int(len(unique_days) * 0.7)
    train_days, test_days = unique_days[:split_idx], unique_days[split_idx:]

    # Create train and test sets based on BookingDate
    train_data = reservations[reservations["BookingDate"].dt.date.isin(train_days)]
    test_data = reservations[reservations["BookingDate"].dt.date.isin(test_days)]

    return train_data, test_data

def get_x_y_sets(tables, train_data, test_data):

    features = ['GuestCount', 'BookingDateDayOfWeek','BookingDateDay', 'BookingDateMonth', 'BookingTime', 'Duration', 'MinCovers', 'MaxCovers']

    X_train, y_train = train_data[features], train_data["TableCode"]
    X_test, y_test = test_data[features], test_data["TableCode"]

    label_encoder = LabelEncoder()
    label_encoder.fit(tables['TableCode'])

    y_train, y_test = label_encoder.transform(y_train), label_encoder.transform(y_test)

    return X_train, X_test, y_train, y_test, label_encoder