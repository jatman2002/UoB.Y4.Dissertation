import csv
import os
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Model:

    def load_data(self, isVal):
        print('LOADING DATA')
        tables = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{self.restaurant_name}-tables.csv')

        if isVal:
            train = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{self.restaurant_name}-train.csv')
            booking_date = pd.to_datetime(train['BookingDate']).dt.date
            unique_days = booking_date.unique()
            val_idx = int(len(unique_days) * 75 / 85)
            val_days = unique_days[val_idx:]
            val = train[booking_date.isin(val_days)]

            val['BookingStartTime'] = (val['BookingStartTime'] - 36000) / (60*15)
            val['Duration'] = val['Duration'] / (60*15)
            val["EndTime"] = val["BookingStartTime"] + val["Duration"]

            return val, tables

        test = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{self.restaurant_name}-test.csv')
        # Adjust some features
        test['BookingStartTime'] = (test['BookingStartTime'] - 36000) / (60*15)
        test['Duration'] = test['Duration'] / (60*15)
        test["EndTime"] = test["BookingStartTime"] + test["Duration"]

        return test, tables
    
    def feature_engineering(self, reservations, use_label_encoder=False):

        booking_date = pd.to_datetime(reservations['BookingDate'])

        reservations['BookingDateDayOfWeek'] = booking_date.dt.dayofweek
        reservations['BookingDateDay'] = booking_date.dt.day
        reservations['BookingDateMonth'] = booking_date.dt.month

        # reservations['BookingStartTime'] = (reservations['BookingStartTime'] - 36000) / (60*15)
        # reservations['Duration'] = reservations['Duration'] / (60*15)
        # reservations["EndTime"] = reservations["BookingStartTime"] + reservations["Duration"]

        # if use_label_encoder:
        #     label_encoder = LabelEncoder()
        #     reservations['TableCode'] = label_encoder.fit_transform(reservations['TableCode'])

    def get_wasted_slots(self, diary):
        min_booking_length = 6
        wasted_count = 0
        for table in diary:
            empty_slots = 0
            for slot in table:
                if slot != 0:
                    if empty_slots < min_booking_length and empty_slots > 0:
                        wasted_count += 1
                    empty_slots = 0
                    continue
                empty_slots += 1

        return wasted_count  
    
    def write_schedule(self, diary, tables, day, num_reservations, num_rejections):
        Path(f'/mnt/fast0/jy894/outputs/{self.file_path}').mkdir(parents=True, exist_ok=True)
        with open(f'/mnt/fast0/jy894/outputs/{self.file_path}/{day.strftime("%Y-%m-%d")}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            output = [['reservations: ', num_reservations, 'rejections:', num_rejections, 'wasted slots', self.get_wasted_slots(diary)],[]]
            for table_code, table in zip(tables, diary):
                table_output = []
                table_output.append(table_code)
                for slot in table:
                    if slot.item() == 0:
                        table_output.append('')
                    else:
                        table_output.append(slot.item())
                output.append(table_output)

            writer.writerows(output)

    def reset_diary(self):
        pass
    

    def run(self):

        booking_date_as_dt = pd.to_datetime(self.test['BookingDate']).dt.date
        unique_days = booking_date_as_dt.unique()

        for i, day in enumerate(unique_days):
            print(f'Looking at day {i} / {len(unique_days)}\t{day=}', end='\r')
            reservations_for_day = self.test.loc[booking_date_as_dt == day]
            rejections = 0

            diary = self.reset_diary()

            for j in range(len(reservations_for_day)):
                reservation = reservations_for_day.iloc[j]
                
                best_table_index = self.find_table(reservation[self.features], diary)
                if best_table_index == -1:
                    rejections += 1
                    continue

                booking_code = int(self.test.loc[self.test.index == reservation.name].iloc[0]['BookingCode'])
                for d in range(int(reservation['Duration'])):
                    diary[best_table_index][int(reservation['BookingStartTime']) + d] = booking_code
                
            self.write_schedule(diary, self.tables['TableCode'].tolist(), day, len(reservations_for_day), rejections)