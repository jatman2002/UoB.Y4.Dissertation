import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import numpy as np
import os
import random
from collections import deque

from RestaurantEnv import RestaurantEnv
from Test import test_predictor

class DqnNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DqnNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, output_size)
        # self.l4 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

class ReplayMemory:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)  # Handles FIFO automatically

    def push(self, e):
        self.buffer.append(e)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def find_table(predictor, reservation, diary, tables):
    
    res_details = torch.tensor(reservation.astype(float).values, dtype=torch.float32, device=device)
    state_details = (diary.flatten() != 0).int()

    with torch.no_grad():
        actions = torch.argsort(predictor(torch.cat((res_details, state_details))), descending=True).tolist()

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
    
def find_y_j_table(predictor, reservations, diaries, tables):
    
    reservations_df = pd.DataFrame(reservations)
    res_details = torch.tensor(reservations_df.astype(float).values, dtype=torch.float32, device=device)

    diaries_tensor = torch.stack(diaries).to(device)
    state_details = (diaries_tensor.flatten(start_dim=1) != 0).int()

    with torch.no_grad():
        actions = torch.argmax(predictor(torch.cat((res_details, state_details), dim=1)), dim=1)

        guest_counts = torch.tensor(reservations_df['GuestCount'].values, device=device)
        min_covers = torch.tensor(tables['MinCovers'].values, device=device)
        max_covers = torch.tensor(tables['MaxCovers'].values, device=device)

        start_times = torch.tensor(reservations_df['BookingStartTime'].astype(int).values, device=device)
        end_times = torch.tensor(reservations_df['EndTime'].astype(int).values, device=device)

        valid_mask = (min_covers[actions] <= guest_counts) & (max_covers[actions] >= guest_counts)

        booking_conflicts = torch.zeros(len(reservations), dtype=torch.bool, device=device)

        # table_diaries = diaries_tensor.gather(1, actions.view(-1,1,1).expand(-1,1,64))
        for b in range(len(reservations)):
            booking_conflicts[b] = torch.any(diaries_tensor[actions[b],0, start_times[b]:end_times[b]] != 0)

        assigned_tables = torch.where(valid_mask & ~booking_conflicts, actions, torch.tensor(-1, device=device))
    return assigned_tables



device = torch.device(
    "cuda:5" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print(f'{device=}')

# Load Data
restaurant_name = 1

print('LOADING DATA')
tables = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-tables.csv')
train = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/MLP-Soft-Encoding/Restaurant-{restaurant_name}-train.csv')

# Adjust some features
train['BookingStartTime'] = (train['BookingStartTime'] - 36000) / (60*15)
train['Duration'] = train['Duration'] / (60*15)
train["EndTime"] = train["BookingStartTime"] + train["Duration"]

features = ['GuestCount', 'BookingStartTime', 'Duration', 'EndTime']

#Create Environment
print('CREATING ENV')
env = RestaurantEnv(tables, device)

# Define networks
print('CREATING NETWORKS')
input_size = len(features) + len(tables)*64
output_size = len(tables)
policy_network = DqnNetwork(input_size, output_size).to(device)
target_network = DqnNetwork(input_size, output_size).to(device)
target_network.load_state_dict(policy_network.state_dict())

optimizer = optim.Adam(policy_network.parameters(), lr=5e-3, amsgrad=True)

# Extra params
epsilon = 0.9
epsilon_decay = 0.9995
gamma = 0.9
# C = 1000
batch_size = 256
TAU = 0.01

# Replay Buffer
print('CREATING REPLAY MEMORY')
memory = ReplayMemory(10000)

booking_date = pd.to_datetime(train['BookingDate']).dt.date
unique_days = booking_date.unique()

actions_taken = {}
for i in range(1,len(tables)+2):
    actions_taken[i] = 0


print('STARTING TRAINING')
for day in unique_days: # a day is an episode
    env.reset(device)
    total_reward = 0
    step = 0

    bookings_on_day = train.loc[booking_date == day]

    training_rejections = 0

    for _, reservation in bookings_on_day.iterrows():

        print(f"Day {day}\t{step+1:>3}/{len(bookings_on_day):<3}", end="\r")
        
        # Get data as tensors for network input
        res_details = torch.tensor(reservation[features].astype(float).values, dtype=torch.float32, device=device)
        current_state = env.state.detach()
        state_details = (current_state.flatten() != 0).int()

        # Explore vs Exploit
        if np.random.rand() < epsilon:
            actions = torch.randperm(len(tables)).to(device)
        else:
            actions = torch.argsort(policy_network(torch.cat((res_details, state_details))), descending=True)

        # Take the action
        action, reward = env.step(actions.tolist(), reservation)

        # extra things to record in logging
        total_reward += reward
        actions_taken[action+1] += 1
        if action == len(tables):
            training_rejections += 1

        # If term state
        if step < len(bookings_on_day)-1:
            next_res = bookings_on_day.iloc[step+1][features]
        else:
            next_res = None

        # Add to replay buffer
        memory.push([current_state, action, reward, env.state.detach(), step == len(bookings_on_day)-1, next_res, reservation[features]])

        step += 1

        # Epsilon decay
        epsilon = max(0.1, epsilon_decay*epsilon)

        if len(memory) < batch_size:
            continue

        # replay sampling
        batch = memory.sample(batch_size)

        s_t, a_t, r_t, s_t_1, term, n_res, res = zip(*batch)

        y_j = torch.tensor(r_t, dtype=torch.float32, device=device)
        x_j = torch.zeros(batch_size, dtype=torch.float32, device=device)

        # load stuff as torch
        inp_res_torch = torch.stack([torch.tensor(r.astype(float).values, dtype=torch.float32, device=device) for r in res])
        n_res_torch = torch.stack([torch.tensor(r.astype(float).values, dtype=torch.float32, device=device) if not r is None else torch.tensor([-1.0,-1.0,-1.0,-1.0], device=device) for r in n_res])

        inp_state_torch = torch.stack([(s.flatten() != 0).int() for s in s_t])
        inp_n_state_torch = torch.stack([(s.flatten() != 0).int() for s in s_t_1])

        # Get current Q value
        policy_actions = find_y_j_table(policy_network, res, s_t, tables)
        rejection = policy_actions == -1
        x_j[~rejection] = policy_network(torch.cat((inp_res_torch[~rejection], inp_state_torch[~rejection]), dim=1)).gather(1, policy_actions[~rejection].unsqueeze(1)).flatten()
        x_j[rejection] = -200

        # Get target Q value
        with torch.no_grad():
            term_states = torch.tensor(term, dtype=torch.bool, device=device)
            policy_n_actions = torch.zeros(batch_size, dtype=torch.int64, device=device)
            policy_n_actions -= 2

            filtered_n_res = tuple(n_res[s] for s in range(len(n_res)) if not term[s])
            filterd_s_t_1 = tuple(s_t_1[s] for s in range(len(s_t_1)) if not term[s])

            policy_n_actions[~term_states] = find_y_j_table(policy_network, filtered_n_res, filterd_s_t_1, tables)
            rejection = policy_n_actions == -1
            invalid_actions = torch.logical_or(term_states, rejection)
            y_j[~invalid_actions] += gamma*(target_network(torch.cat((n_res_torch[~invalid_actions], inp_n_state_torch[~invalid_actions]), dim=1)).gather(1, policy_n_actions[~invalid_actions].unsqueeze(1))).flatten()
            y_j[rejection] -= 200
            
        # Update policy network
        criterion = nn.HuberLoss()
        loss = criterion(x_j, y_j)

        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_network.parameters(), 100.0)
        optimizer.step()

        # Soft update of target network
        for target_param, policy_param in zip(target_network.parameters(), policy_network.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)

    print(f"\nDay {day}\tproportional_reward={total_reward/len(bookings_on_day):<18}\trejections = {training_rejections}")

print()
print(actions_taken)
print()

torch.save(policy_network.state_dict(), f'{os.getcwd()}/models/DQN3.pt')

test_data = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-test.csv')
test_data['BookingStartTime'] = (test_data['BookingStartTime'] - 36000) / (60*15)
test_data['Duration'] = test_data['Duration'] / (60*15)
test_data["EndTime"] = test_data["BookingStartTime"] + test_data["Duration"]

print()
test_predictor(f'Restaurant-{restaurant_name}/DQN3', test_data, tables, policy_network, find_table, device, features)