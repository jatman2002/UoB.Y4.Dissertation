import os
import pandas as pd
from .KerasModel import KerasModel
import numpy as np
import keras

class MLP3_4(KerasModel):
    def __init__(self, restaurant_name, name, isVal):
        super().__init__(restaurant_name, name, isVal)

    def get_model_input(self, res_details, diary):
        state_details = (diary.flatten() != 0).astype(int)
        model_input = np.concatenate((res_details, state_details)).reshape(1,-1)
        return model_input
    
    def load_data(self, isVal):
        print('LOADING DATA')
        tables = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{self.restaurant_name}-tables.csv')

        if isVal:
            train = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/MLP-State/Restaurant-{self.restaurant_name}-train.csv')
            booking_date = pd.to_datetime(train['BookingDate']).dt.date
            unique_days = booking_date.unique()
            val_idx = int(len(unique_days) * 75 / 85)
            val_days = unique_days[val_idx:]
            val = train[booking_date.isin(val_days)]

            return val, tables

        test = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/MLP-State/Restaurant-{self.restaurant_name}-test.csv')
        # Adjust some features
        test['BookingStartTime'] = (test['BookingStartTime'] - 36000) / (60*15)
        test['Duration'] = test['Duration'] / (60*15)
        test["EndTime"] = test["BookingStartTime"] + test["Duration"]

        return test, tables
    
class MLP3(MLP3_4):
    def __init__(self, restaurant_name, isVal):
        super().__init__(restaurant_name, 'MLP3', isVal)
            
class MLP4(MLP3_4):
    def __init__(self, restaurant_name, isVal):
        super().__init__(restaurant_name, 'MLP4', isVal)

class MLP(MLP3_4):
    def __init__(self, restaurant_name, isVal, model_name):
        self.model_name = model_name
        self.file_path = f'Restaurant-{restaurant_name}/MLP/{model_name}'
        super().__init__(restaurant_name, 'MLP', isVal)

    def load_model(self):
        self.model = keras.saving.load_model(f'/mnt/fast0/jy894/models/{self.name}/grid/{self.name}-{self.model_name}.keras')