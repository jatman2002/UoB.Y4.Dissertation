# imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .helper.dataset import get_data

import pickle
from pathlib import Path

class RF:
    def __init__(self, restaurant_name, gpu):
        self.name = 'RF'
        self.restaurant_name = restaurant_name

    def find_table(self, predictor, reservation, diary, tables):

        probabilities = predictor.predict_proba(pd.DataFrame([reservation]))[0]
        order_of_tables = np.argsort(probabilities)[::-1]

        best_table_index = -1

        # in order of probability, find the first one that fits
        for t in order_of_tables:
            best_table = tables.iloc[t]

            # ignore where prob is 0 i.e. the classifier will never choose it
            if probabilities[t] <= 0.:
                continue

            # size constraint
            if best_table['MinCovers'] > reservation['GuestCount'] or best_table['MaxCovers'] < reservation['GuestCount']:
                continue

            # 2. time constraint
            if np.all(diary[t][int(reservation['BookingStartTime']):int(reservation['EndTime'])] != [None]*int(reservation['Duration'])):
                continue

            return t

        return best_table_index
    
    def load_data(self):
        
        # load data from csv (raw data)
        X, y, test_data, features, tables = get_data(self.restaurant_name, use_label_encoder=False)

        booking_date = pd.to_datetime(X['BookingDate']).dt.date
        unique_days = booking_date.unique()

        val_idx = int(len(unique_days) * 70 / 85)

        train_days, val_days = unique_days[:val_idx], unique_days[val_idx:]

        X_train = X[booking_date.isin(train_days)]
        X_val = X[booking_date.isin(val_days)]

        y_train = y[:len(X_train)]
        y_val = y[len(X_train):]

        X_train = X_train[features]

        return X_train, y_train, X_val, y_val, tables, features


    def run(self):

        # load data from csv (raw data)
        X_train, y_train, X_val, y_val, tables, features = self.load_data()
        
        # fit the RF
        print('TRAINING THE RANDOM FOREST CLASSIFIER')
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
        )
        classifier.fit(X_train, y_train)

        Path(f'/mnt/fast0/jy894/models/{self.name}/models').mkdir(parents=True, exist_ok=True)

        with open(f'/mnt/fast0/jy894/models/{self.name}/models/{self.name}-R-{self.restaurant_name}.pkl', 'wb') as f:
            pickle.dump(classifier, f)