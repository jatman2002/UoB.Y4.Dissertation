# imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

from helper.Test import test_predictor, get_wasted_slots
from helper.dataset import get_data

def find_table(predictor, reservation, diary, tables):

    probabilities = predictor.predict_proba(pd.DataFrame([reservation]))[0]
    order_of_tables = np.argsort(probabilities)[::-1]

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

    return -1


def custom_scorer(estimator, X, tables, features, get_best_table):

    booking_date_as_dt = pd.to_datetime(X['BookingDate']).dt.date
    unique_days = booking_date_as_dt.unique()

    total_rejections = 0
    total_penalty = 0

    for i, day in enumerate(unique_days):
        print(f'Looking at day {i} / {len(unique_days)}', end='\r')
        reservations_for_day = X.loc[booking_date_as_dt == day]
        diary = [[None] * 64 for _ in range(len(tables))]  # 64 time slots

        for _, reservation in reservations_for_day.iterrows():
            best_table_index = get_best_table(estimator, reservation[features], diary, tables)
            if best_table_index == -1:
                total_rejections += 1
                continue

            booking_code = str(reservation['BookingCode'])
            start_time = int(reservation['BookingStartTime'])
            duration = int(reservation['Duration'])

            # Assign reservation in diary
            for t in range(duration):
                diary[best_table_index][start_time + t] = booking_code

            # Calculate empty slots before & after
            # empty_before = sum(1 for t in range(start_time - 6, start_time) if 0 <= t < 64 and diary[best_table_index][t] is None)
            # empty_after = sum(1 for t in range(start_time + duration, start_time + duration + 6) if 0 <= t < 64 and diary[best_table_index][t] is None)

            # Apply penalty for inefficiency
            total_penalty += get_wasted_slots(diary)

    # Compute final score
    score = (total_rejections* (10**4) + total_penalty)  # Weighted penalty
    return score


def run(restaurant_name):

    # load data from csv (raw data)
    X, y, test_data, features, tables = get_data(restaurant_name, use_label_encoder=False)

    booking_date = pd.to_datetime(X['BookingDate']).dt.date
    unique_days = booking_date.unique()

    val_idx = int(len(unique_days) * 70 / 85)

    train_days, val_days = unique_days[:val_idx], unique_days[val_idx:]

    X_train = X[booking_date.isin(train_days)]
    X_val = X[booking_date.isin(val_days)]

    y_train = y[:len(X_train)]
    y_val = y[len(X_train):]

    X_train = X_train[features]

    param_grid = {
        'n_estimators': [1, 5, 10, 50, 100, 200],
        'max_depth': [10, 20, None],
        # 'min_samples_split': [2, 5],
        # 'min_samples_leaf': [1, 2],
        # 'max_features': ['sqrt', 'log2'],
        'random_state': [42]
    }

    param_grid = ParameterGrid(param_grid)
    
    # fit the RF
    print('TRAINING THE RANDOM FOREST CLASSIFIER\n')

    best_score = np.inf
    best_p = None
    best_classifier = None

    for p in param_grid:
        print(f'TESTING PARAMETERS {p}')
        classifier = RandomForestClassifier()
        classifier.set_params(**p)
        classifier.fit(X_train, y_train)

        score = custom_scorer(classifier, X_val, tables, features, find_table)
        print(f'{score=}')
        if score < best_score:
            best_score = score
            best_p = p
            best_classifier = classifier

    print(f'BEST PARAMS - {best_p}')


    # test RF on data
    print('TIME TO TEST THIS THING ~~0_0~~\n')
    test_predictor(f'Restaurant-{restaurant_name}/RFGridSearch', test_data, tables, best_classifier, find_table, features)
    print()
    print('DONE!')