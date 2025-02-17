# imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder

from Test import test_predictor


def find_table(predictor, reservation, diary):

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


# load data from csv (raw data)

restaurant_name = '1'

print('LOADING DATA FROM CSV')
reservations = pd.read_csv(f'C:/git/UoB.Y4.Dissertation/src/Restaurant-{restaurant_name}/output.csv')
tables = pd.read_csv(f'C:/git/UoB.Y4.Dissertation/src/Restaurant-{restaurant_name}/tables.csv')
print('DATA LOADED')

# feature 
print('MESSING AROUND WITH FEATURES')
# f1 = lambda t: f'is{t["TableCode"]}Free'
# f2 = lambda t: f'{t["TableCode"]}MinCovers'
# f3 = lambda t: f'{t["TableCode"]}MaxCovers'


# features_to_drop= [f(t) for _,t in tables.iterrows() for f in (f1,f2,f3)]
# reservations = reservations.drop(columns=features_to_drop, axis=1)

booking_date = pd.to_datetime(reservations['BookingDate'])

reservations['BookingDateDayOfWeek'] = booking_date.dt.dayofweek
reservations['BookingDateDay'] = booking_date.dt.day
reservations['BookingDateMonth'] = booking_date.dt.month

reservations['BookingTime'] = (reservations['BookingTime'] - 36000) / (60*15)
reservations['Duration'] = reservations['Duration'] / (60*15)
reservations["EndTime"] = reservations["BookingTime"] + reservations["Duration"]

label_encoder = LabelEncoder()
reservations['TableCode'] = label_encoder.fit_transform(reservations['TableCode'])


# Train/Test split
print('SPLITTING DATA INTO TRAIN AND TEST')
# Get unique days and shuffle them
unique_days = booking_date.dt.date.unique()
# np.random.shuffle(unique_days)

# Split 70% for training, 30% for testing
split_idx = int(len(unique_days) * 0.7)
train_days, test_days = unique_days[:split_idx], unique_days[split_idx:]

# Create train and test sets based on BookingDate
train_data = reservations[booking_date.dt.date.isin(train_days)]
test_data = reservations[booking_date.dt.date.isin(test_days)]

features = ['GuestCount', 'BookingDateDayOfWeek', 'BookingDateMonth', 'BookingTime', 'Duration', 'EndTime']
# [features.append(f(t)) for _,t in tables.iterrows() for f in (f1,f2,f3)]
X_train, y_train = torch.tensor(train_data[features].values, dtype=torch.float32), torch.tensor(train_data["TableCode"].values, dtype=torch.long)

# # fit the RF
print('TRAINING THE MLP CLASSIFIER')
# classifier = RandomForestClassifier(n_estimators=200, max_depth=10)
# classifier.fit(X_train, y_train)

inp = len(features)
hidden_1 = 6 + (np.abs(len(tables) - 6)//4)
hidden_2 = 6 + ((np.abs(len(tables) - 6)*2)//4)
hidden_3 = 6 + ((np.abs(len(tables) - 6)*3)//4)
output = len(tables)

# Create MLP
model = nn.Sequential(
    nn.Linear(inp, hidden_1),
    nn.ReLU(),
    nn.Linear(hidden_1, hidden_2),
    nn.ReLU(),
    nn.Linear(hidden_2, hidden_3),
    nn.ReLU(),
    nn.Linear(hidden_3, output),
    nn.Softmax(dim=0)
)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 250

for n in range(num_epochs):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# pred = model(torch.tensor(test_data.iloc[0][features].astype(float).values, dtype=torch.float32))

# print(pred)
# test RF on data
print('TIME TO TEST THIS THING ~~0_0~~\n')
test_predictor(f'Restaurant-{restaurant_name}/MLP', test_data, tables, model, find_table, features)
print()
print('DONE!')