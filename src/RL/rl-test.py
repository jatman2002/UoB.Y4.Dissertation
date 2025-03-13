import torch
import torch.nn as nn

import os
import pandas as pd
import numpy as np

from RestaurantEnv import RestaurantEnv
from Test import test_predictor

def find_table(predictor, reservation, diary, tables):

    # probabilities = classifier.predict_proba(pd.DataFrame([reservation]))[0]

    res_details = torch.tensor(reservation.astype(float).values, dtype=torch.float32)
    state_details = diary.flatten()

    probabilities = predictor(torch.cat((res_details, state_details))).detach().numpy()
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
        if np.all(diary[t][int(reservation['BookingStartTime']):int(reservation['BookingEndTime'])] != [None]*int(reservation['Duration'])):
            continue

        return t

    return best_table_index

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)
    

restaurant_name = 1
    

tables = tables = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-tables.csv')
train = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/MLP-Soft-Encoding/Restaurant-{restaurant_name}-train.csv')

features = ['GuestCount', 'BookingStartTime', 'Duration', 'BookingEndTime']

learning_rate = 0.01
gamma = 0.99  # Discount factor

env = RestaurantEnv(tables)
policy = PolicyNetwork(len(features)+len(tables)*64, len(tables))
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

booking_date = pd.to_datetime(train['BookingDate']).dt.date
unique_days = booking_date.unique()

for _ in range(5):
    for day in unique_days: # a day is an episode
        env.reset()
        total_reward = 0
        step = 0

        for _, reservation in train.loc[booking_date == day].iterrows():

            res_details = torch.tensor(reservation[features].astype(float).values, dtype=torch.float32)
            state_details = env.state.flatten()

            action_probs = policy(torch.cat((res_details, state_details)))
            action = torch.multinomial(action_probs, 1).item()
            reward = env.step(action, reservation)

            log_prob = torch.log(action_probs[action])
            discounted_reward = reward * (gamma ** step)
            loss = -log_prob * discounted_reward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += reward
            step += 1

        print(f"Day {day}, Total Reward: {total_reward}")


test_data = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-test.csv')


test_predictor(f'Restaurant-{restaurant_name}/RL-test', test_data, tables, policy, find_table, features)