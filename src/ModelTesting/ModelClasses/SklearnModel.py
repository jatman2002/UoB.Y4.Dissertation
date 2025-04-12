import sklearn

import os
import numpy as np

import pandas as pd

import pickle

from .Model import Model

class SklearnModel(Model):
    def __init__(self, restaurant_name, name, isVal):
        self.name = name
        self.restaurant_name = restaurant_name
        self.features = ['GuestCount', 'BookingDateDayOfWeek', 'BookingDateMonth', 'BookingStartTime', 'Duration', 'EndTime']
        self.test, self.tables = self.load_data(False)
        self.feature_engineering(self.test)
        self.model = None
        self.file_path = f'Restaurant-{restaurant_name}/{name}'
        self.load_model()

    def load_model(self):
        pickle_path = f'/mnt/fast0/jy894/models/{self.name}/models/{self.name}-R-{self.restaurant_name}.pkl'
        with open(pickle_path, 'rb') as file:
            self.model = pickle.load(file)

    def find_table(self, reservation, diary):

        probabilities = self.model.predict_proba(pd.DataFrame([reservation]))[0]
        order_of_tables = np.argsort(probabilities)[::-1]

        best_table_index = -1

        # in order of probability, find the first one that fits
        for t in order_of_tables:
            best_table = self.tables.iloc[t]

            # ignore where prob is 0 i.e. the classifier will never choose it
            if probabilities[t] <= 0.:
                continue

            # size constraint
            if best_table['MinCovers'] > reservation['GuestCount'] or best_table['MaxCovers'] < reservation['GuestCount']:
                continue

            # 2. time constraint
            if np.any(diary[t][int(reservation['BookingStartTime']):int(reservation['EndTime'])] != 0):
                continue

            return t

        return best_table_index
    
    def reset_diary(self):
        return np.zeros((len(self.tables), 64))
    
class RF(SklearnModel):
    def __init__(self, restaurant_name, isVal):
        super().__init__(restaurant_name, 'RF', isVal)

class LR(SklearnModel):
    def __init__(self, restaurant_name, isVal):
        super().__init__(restaurant_name, 'LR', isVal)