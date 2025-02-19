# imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from typing import Optional
from torch import Tensor

from sklearn.preprocessing import LabelEncoder

from Test import test_predictor
from dataset import get_data

class CustomLoss(nn.CrossEntropyLoss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0):
        super().__init__()

    def forward(self, inp: Tensor, target: Tensor, score):
        CEL = super().forward(inp, target)

        return CEL + score*score


def find_table(predictor, reservation, diary, tables):

    # probabilities = classifier.predict_proba(pd.DataFrame([reservation]))[0]
    probabilities = predictor(torch.tensor(reservation.astype(float).values, dtype=torch.float32)).detach().numpy()
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
        if np.all(diary[t][int(reservation['BookingTime']):int(reservation['EndTime'])] != [None]*int(reservation['Duration'])):
            continue

        return t

    return best_table_index


def get_wasted_slots(diary):
    min_booking_length = 6
    total_wasted_slots = 0
    for table in diary:
        wasted_slots = 0
        for slot in table:
            if slot == None:
                wasted_slots += 1
            else:
                total_wasted_slots += wasted_slots % min_booking_length
                wasted_slots = 0

    return total_wasted_slots

def run(restaurant_name):

    #------------------------------------------------------------------------------------------------------------------------------------

    # LOAD DATA

    train_data, test_data, features, restaurant_name, tables = get_data(restaurant_name, use_label_encoder=True)
    X_train = train_data[features]
    y_train = train_data['TableCode']


    start_time_idx = features.index('BookingTime')
    duration_idx = features.index('Duration')
    end_time_idx = features.index('EndTime')
    guest_count_idx = features.index('GuestCount')

    # X_train_torch, y_train_torch = torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.long)

    #------------------------------------------------------------------------------------------------------------------------------------

    # TRAIN MODEL

    print('TRAINING THE MLP CLASSIFIER')

    inp = len(features)
    hidden_1 = 6 + (np.abs(len(tables) - 6)//2)
    # hidden_2 = 6 + ((np.abs(len(tables) - 6)*2)//4)
    # hidden_3 = 6 + ((np.abs(len(tables) - 6)*3)//4)
    output = len(tables)

    # Create MLP
    # model = nn.Sequential(
    #     nn.Linear(inp, hidden_1),
    #     nn.ReLU(),
    #     nn.Linear(hidden_1, hidden_2),
    #     nn.ReLU(),
    #     nn.Linear(hidden_2, hidden_3),
    #     nn.ReLU(),
    #     nn.Linear(hidden_3, output),
    #     nn.Softmax(dim=0)
    # )

    model = nn.Sequential(
        nn.Linear(inp, hidden_1),
        nn.ReLU(),
        nn.Linear(hidden_1, output),
        nn.Softmax(dim=0)
    )

    loss_fn = CustomLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 400

    unique_days = pd.to_datetime(train_data['BookingDate']).dt.date.unique()

    for n in range(num_epochs):

        total_score = 0

        for day in unique_days:

            indicies = train_data.loc[train_data['BookingDate'] == day.strftime('%Y-%m-%d')].index
            X_train_day = X_train.iloc[indicies]
            y_train_day = y_train.iloc[indicies]

            X_train_day, y_train_day = torch.tensor(X_train_day.values, dtype=torch.float32), torch.tensor(y_train_day.values, dtype=torch.long)

            y_pred = model(X_train_day)

        #====================================================================================================================================

            # target_diary = [[None] * 64 for _ in range(len(tables))]
            model_diary = [[None] * 64 for _ in range(len(tables))]

            # for booking, target, pred in zip(X_train_day.tolist(), y_train_day.tolist(), y_pred.tolist()):
            for booking, pred in zip(X_train_day.tolist(), y_pred.tolist()):

                # for i in range(int(booking[duration_idx])):
                #     target_diary[target][int(booking[start_time_idx]) + i] = 1 # occupied


                order_of_tables = np.argsort(pred)[::-1]
                for t in order_of_tables:
                    table = tables.iloc[t]

                    # size constraint
                    if table['MinCovers'] > booking[guest_count_idx] or table['MaxCovers'] < booking[guest_count_idx]:
                        continue

                    # 2. time constraint
                    if np.all(model_diary[t][int(booking[start_time_idx]):int(booking[end_time_idx])] != [None]*int(booking[duration_idx])):
                        continue

                    for i in range(int(booking[duration_idx])):
                        model_diary[t][int(booking[start_time_idx]) + i] = 1 # occupied
                    break

            # target_wasted_slots = get_wasted_slots(target_diary)
            model_wasted_slots = get_wasted_slots(model_diary)

            total_score += model_wasted_slots #- target_wasted_slots

            print(f'epoch_num = {n}\tday = {day}')#\t{target_wasted_slots=}\t{model_wasted_slots=}')


        #====================================================================================================================================

        loss = loss_fn(y_pred, y_train_day, total_score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #------------------------------------------------------------------------------------------------------------------------------------

    # TEST MODEL

    print('TIME TO TEST THIS THING ~~0_0~~\n')
    test_predictor(f'Restaurant-{restaurant_name}/MLP', test_data, tables, model, find_table, features)
    print()
    print('DONE!')


# run('1')