import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import numpy as np
import os
import random

from RestaurantEnv import RestaurantEnv
from Test import test_predictor

class DqnNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DqnNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, input_size-(input_size-output_size)//5)
        self.l2 = nn.Linear(input_size-(input_size-output_size)//5, input_size-2*(input_size-output_size)//5)
        self.l3 = nn.Linear(input_size-2*(input_size-output_size)//5, input_size-3*(input_size-output_size)//5)
        self.l4 = nn.Linear(input_size-3*(input_size-output_size)//5, input_size-4*(input_size-output_size)//5)
        self.l5 = nn.Linear(input_size-4*(input_size-output_size)//5, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)
    

class ReplayMemory:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = [] # [state_t, action_t, reward_t, state_t+1, is_terminal, res_details]

    def push(self, e):
        if len(self.buffer) == self.buffer_size:
            self.buffer = self.buffer[1:]
        self.buffer.append(e)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def find_table(predictor, reservation, diary, tables):
    
    res_details = torch.tensor(reservation.astype(float).values, dtype=torch.float32, device=device)
    state_details = (diary.flatten() != 0).int()

    with torch.no_grad():
        actions = torch.argsort(predictor(torch.cat((res_details, state_details)))).tolist()

        for a_i in range(len(actions)):
        
            start = int(reservation['BookingStartTime'])
            end = int(reservation['EndTime'])

            a = actions[a_i]

            #heavily penalise incorrect tables
            if tables.iloc[a]['MinCovers'] > reservation['GuestCount']:
                continue
            if tables.iloc[a]['MaxCovers'] < reservation['GuestCount']:
                continue
            if torch.any(diary[a][start:end] != 0).item():
                continue
            
            return a

        return -1


    

device = torch.device(
    "cuda:1" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# print(f'{device=}\t{torch.cuda.current_device()=}')

# Load Data
restaurant_name = 1

tables = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-tables.csv')
train = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/MLP-Soft-Encoding/Restaurant-{restaurant_name}-train.csv')

# Adjust some features
train['BookingStartTime'] = (train['BookingStartTime'] - 36000) / (60*15)
train['Duration'] = train['Duration'] / (60*15)
train["EndTime"] = train["BookingStartTime"] + train["Duration"]

features = ['GuestCount', 'BookingStartTime', 'Duration', 'EndTime']

#Create Environment
env = RestaurantEnv(tables, device)

# Define networks
input_size = len(features) + len(tables)*64
output_size = len(tables)
policy_network = DqnNetwork(input_size, output_size).to(device)
target_network = DqnNetwork(input_size, output_size).to(device)
target_network.load_state_dict(policy_network.state_dict())

optimizer = optim.Adam(policy_network.parameters(), lr=1e-4, amsgrad=True)

# Extra params
epsilon = 0.9
gamma = 0.9
C = 10
batch_size = 5

# Replay Buffer
memory = ReplayMemory(1000)

booking_date = pd.to_datetime(train['BookingDate']).dt.date
unique_days = booking_date.unique()

for day in unique_days: # a day is an episode
    env.reset(device)
    total_reward = 0
    step = 0

    bookings_on_day = train.loc[booking_date == day]

    for _, reservation in bookings_on_day.iterrows():

        print(f"Day {day}\t{step+1:>3}/{len(bookings_on_day):<3}", end="\r")
        
        res_details = torch.tensor(reservation[features].astype(float).values, dtype=torch.float32, device=device)
        current_state = env.state.detach()
        state_details = (current_state.flatten() != 0).int()

        # Explore vs Exploit
        if np.random.rand() < epsilon:
            actions = torch.randperm(len(tables)).to(device)
        else:
            actions = torch.argsort(policy_network(torch.cat((res_details, state_details))))

        # Take the action
        action, reward = env.step(actions.tolist(), reservation)
        total_reward += reward

        if step < len(bookings_on_day)-1:
            next_res = bookings_on_day.iloc[step+1][features]
        else:
            next_res = None

        # Add to replay buffer
        memory.push([current_state, action, reward, env.state.detach(), step == len(bookings_on_day)-1, next_res, reservation[features]])

        step += 1

        # Epsilon decay
        epsilon = max(0.1, 0.9999*epsilon)

        if len(memory) < batch_size:
            continue

        # replay sampling
        batch = memory.sample(batch_size)

        s_t, a_t, r_t, s_t_1, term, n_res, res = zip(*batch)

        y_j = torch.tensor(r_t, dtype=torch.float32, device=device)
        x_j = torch.zeros(batch_size, dtype=torch.float32, device=device)

        inp_res_torch = torch.stack([torch.tensor(r.astype(float).values, dtype=torch.float32, device=device) for r in res])
        n_res_torch = torch.stack([torch.tensor(r.astype(float).values, dtype=torch.float32, device=device) if not r is None else torch.tensor([-1.0,-1.0,-1.0,-1.0], device=device) for r in n_res])

        inp_state_torch = torch.stack([(s.flatten() != 0).int() for s in s_t])
        inp_n_state_torch = torch.stack([(s.flatten() != 0).int() for s in s_t_1])

        policy_actions = torch.tensor([find_table(policy_network, r, s, tables) for s, r in zip(s_t, res)], device=device)
        rejection = policy_actions == -1
        x_j[~rejection] = policy_network(torch.cat((inp_res_torch, inp_state_torch), dim=1)).gather(1, policy_actions.unsqueeze(1)).flatten()


        term_states = torch.tensor([term for _, _, _, _, term, _, _ in batch], device=device)
        policy_n_actions = torch.tensor([find_table(policy_network, n_r, n_s, tables) if not n_r is None else -1 for n_r, n_s in zip(n_res, s_t_1)], device=device)
        rejection = policy_n_actions == -1
        invalid_actions = torch.logical_or(term_states, rejection)
        y_j[~invalid_actions] += gamma*(target_network(torch.cat((n_res_torch[~invalid_actions], inp_n_state_torch[~invalid_actions]), dim=1)).gather(1, policy_n_actions[~rejection].unsqueeze(1))).flatten()

        # y_j = []
        # x_j = []
        # for (s_t, _, r, s_t_1, term, n_res, res) in batch:
        #     policy_action = find_table(policy_network, res, s_t, tables)
        #     if policy_action == -1: # rejection
        #         x_j.append(torch.tensor(0, dtype=torch.float32, device=device))
        #     else:
        #         inp_res = torch.tensor(res.astype(float).values, dtype=torch.float32, device=device)
        #         inp_state = (s_t.flatten() != 0).int()
        #         x_j.append(policy_network(torch.cat((inp_res, inp_state)))[policy_action])

        #     if term:
        #         y_j.append(torch.tensor(r, dtype=torch.float32, device=device))
        #     else:
        #         policy_action_n_res = find_table(policy_network, n_res, s_t_1, tables)

        #         if policy_action_n_res == -1: # if it rejects
        #             y_j.append(torch.tensor(r, dtype=torch.float32, device=device))
        #         else:
        #             inp_n_res = torch.tensor(n_res.astype(float).values, dtype=torch.float32, device=device)
        #             inp_n_state = (s_t_1.flatten() != 0).int()
                    # y_j.append(r + gamma*(target_network(torch.cat((inp_n_res, inp_n_state)))[policy_action_n_res]))

        # Update policy network
        criterion = nn.HuberLoss()
        loss = criterion(x_j, y_j)

        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_network.parameters(), 1.0)
        optimizer.step()

        # Update target network every C steps
        if step % C == 0:
            target_network.load_state_dict(policy_network.state_dict())

    print(f"\nDay {day}\tproportional_reward={total_reward/len(bookings_on_day):>5}")


test_data = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-test.csv')
test_data['BookingStartTime'] = (test_data['BookingStartTime'] - 36000) / (60*15)
test_data['Duration'] = test_data['Duration'] / (60*15)
test_data["EndTime"] = test_data["BookingStartTime"] + test_data["Duration"]

print()
test_predictor(f'Restaurant-{restaurant_name}/DQN', test_data, tables, policy_network, find_table, device, features)