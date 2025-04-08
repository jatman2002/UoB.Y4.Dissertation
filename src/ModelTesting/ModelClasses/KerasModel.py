import keras

import os
import numpy as np

from .Model import Model

class KerasModel(Model):
    def __init__(self, restaurant_name, name, isVal):
        self.name = name
        self.restaurant_name = restaurant_name
        self.features = ['GuestCount', 'BookingDateDayOfWeek', 'BookingDateMonth', 'BookingStartTime', 'Duration', 'EndTime']
        self.test, self.tables = self.load_data(isVal)
        self.feature_engineering(self.test)
        self.model = None
        self.load_model()
        self.file_path = f'Restaurant-{restaurant_name}/{name}'

    def load_model(self):
        self.model = keras.saving.load_model(f'{os.getcwd()}/models/{self.name}/models/{self.name}-R-{self.restaurant_name}.keras')


    def find_table(self, reservation, diary):

        res_details = reservation.astype(float).values
        # model_input = res_details.reshape(1,-1)
        model_input = self.get_model_input(res_details, diary)
        probabilities = self.model(model_input, training=False)[0]
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
            if np.all(diary[t][int(reservation['BookingStartTime']):int(reservation['EndTime'])] != 0):
                continue

            return t

        return best_table_index
    
    def reset_diary(self):
        return np.zeros((len(self.tables), 64))