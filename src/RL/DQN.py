import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import numpy as np
import os
import random
from collections import deque
import pickle

from .RestaurantEnv import RestaurantEnv

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
    
class DQN:
    def __init__(self, restaurant_name, gpu=0):
        print('-'*5, 'DQN', '-'*5)
        self.restaurant_name = restaurant_name

        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

        # Extra params
        self.epsilon = 0.9
        self.epsilon_decay = 0.9995
        self.gamma = 0.9
        self.batch_size = 256
        self.TAU = 0.01

        print('LOADING DATA')
        self.train, self.tables, self.features = self.load_data()

        print('CREATING ENV')
        self.env = RestaurantEnv(self.tables, self.device)

        # Define networks
        print('CREATING NETWORKS')
        input_size = len(self.features) + len(self.tables)*64
        output_size = len(self.tables)
        self.policy_network = DqnNetwork(input_size, output_size).to(self.device)
        self.target_network = DqnNetwork(input_size, output_size).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=1e-3, amsgrad=True)

        # Replay Buffer
        print('CREATING REPLAY MEMORY')
        self.memory = ReplayMemory(30000)
    
    def find_y_j_table(self, predictor, reservations, diaries, tables):
        
        reservations_df = pd.DataFrame(reservations)
        res_details = torch.tensor(reservations_df.astype(float).values, dtype=torch.float32, device=self.device)

        diaries_tensor = torch.stack(diaries).to(self.device)
        state_details = (diaries_tensor.flatten(start_dim=1) != 0).to(self.device).int()

        with torch.no_grad():
            actions = torch.argmax(predictor(torch.cat((res_details, state_details), dim=1)).to(self.device), dim=1).to(self.device)

            guest_counts = torch.tensor(reservations_df['GuestCount'].values, device=self.device)
            min_covers = torch.tensor(tables['MinCovers'].values, device=self.device)
            max_covers = torch.tensor(tables['MaxCovers'].values, device=self.device)

            start_times = torch.tensor(reservations_df['BookingStartTime'].astype(int).values, device=self.device)
            end_times = torch.tensor(reservations_df['EndTime'].astype(int).values, device=self.device)

            valid_mask = (min_covers[actions] <= guest_counts) & (max_covers[actions] >= guest_counts).to(self.device)

            booking_conflicts = torch.zeros(len(reservations), dtype=torch.bool, device=self.device)

            # table_diaries = diaries_tensor.gather(1, actions.view(-1,1,1).expand(-1,1,64))
            for b in range(len(reservations)):
                booking_conflicts[b] = torch.any(diaries_tensor[actions[b],0, start_times[b]:end_times[b]] != 0).to(self.device)

            assigned_tables = torch.where(valid_mask & ~booking_conflicts, actions, torch.tensor(-1, device=self.device)).to(self.device)
        return assigned_tables

    def load_data(self):
        print('LOADING DATA')
        tables = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{self.restaurant_name}-tables.csv')
        train = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/MLP-Soft-Encoding/Restaurant-{self.restaurant_name}-train.csv')

        # Adjust some features
        train['BookingStartTime'] = (train['BookingStartTime'] - 36000) / (60*15)
        train['Duration'] = train['Duration'] / (60*15)
        train["EndTime"] = train["BookingStartTime"] + train["Duration"]

        features = ['GuestCount', 'BookingStartTime', 'Duration', 'EndTime']

        return train, tables, features
    

    def update_networks(self):
        batch = self.memory.sample(self.batch_size)

        s_t, a_t, r_t, s_t_1, term, n_res, res = zip(*batch)

        y_j = torch.tensor(r_t, dtype=torch.float32, device=self.device)
        x_j = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)

        a_t_torch = torch.tensor(a_t, device=self.device)

        # load stuff as torch
        inp_res_torch = torch.stack([torch.tensor(r.astype(float).values, dtype=torch.float32, device=self.device) for r in res]).to(self.device)
        n_res_torch = torch.stack([torch.tensor(r.astype(float).values, dtype=torch.float32, device=self.device) if not r is None else torch.tensor([-1.0,-1.0,-1.0,-1.0], device=self.device) for r in n_res]).to(self.device)

        inp_state_torch = torch.stack([(s.flatten() != 0).int() for s in s_t]).to(self.device)
        inp_n_state_torch = torch.stack([(s.flatten() != 0).int() for s in s_t_1]).to(self.device)

        # Get current Q value
        rejection = a_t_torch == len(self.tables)
        x_j[~rejection] = self.policy_network(torch.cat((inp_res_torch[~rejection], inp_state_torch[~rejection]), dim=1)).gather(1, a_t_torch[~rejection].unsqueeze(1)).flatten()
        x_j[rejection] = -200

        # Get target Q value
        with torch.no_grad():
            term_states = torch.tensor(term, dtype=torch.bool, device=self.device)
            policy_n_actions = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
            policy_n_actions -= 2

            filtered_n_res = tuple(n_res[s] for s in range(len(n_res)) if not term[s])
            filterd_s_t_1 = tuple(s_t_1[s] for s in range(len(s_t_1)) if not term[s])

            policy_n_actions[~term_states] = self.find_y_j_table(self.policy_network, filtered_n_res, filterd_s_t_1, self.tables)
            rejection = policy_n_actions == -1
            invalid_actions = torch.logical_or(term_states, rejection).to(self.device)
            y_j[~invalid_actions] += self.gamma*(self.target_network(torch.cat((n_res_torch[~invalid_actions], inp_n_state_torch[~invalid_actions]), dim=1)).gather(1, policy_n_actions[~invalid_actions].unsqueeze(1))).flatten()
            y_j[rejection] -= 200
            
        # Update policy network
        criterion = nn.HuberLoss()
        loss = criterion(x_j, y_j)

        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100.0)
        self.optimizer.step()

        # Soft update of target network
        for target_param, policy_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(self.TAU * policy_param.data + (1.0 - self.TAU) * target_param.data)
    

    def run(self):

        booking_date = pd.to_datetime(self.train['BookingDate']).dt.date
        unique_days = booking_date.unique()

        actions_taken = {}
        for i in range(1,len(self.tables)+2):
            actions_taken[i] = 0

        trajectories = {}


        print('STARTING TRAINING')
        for day in unique_days: # a day is an episode
            self.env.reset(self.device)
            total_reward = 0
            step = 0

            bookings_on_day = self.train.loc[booking_date == day]

            training_rejections = 0
            trajectories[day] = []

            for _, reservation in bookings_on_day.iterrows():

                print_string = f"Day {day}\t{step+1:>3}/{len(bookings_on_day):<3}",
                
                # Get data as tensors for network input
                res_details = torch.tensor(reservation[self.features].astype(float).values, dtype=torch.float32, device=self.device)
                current_state = self.env.state.detach()
                state_details = (current_state.flatten() != 0).to(self.device).int()

                # Explore vs Exploit
                if np.random.rand() < self.epsilon:
                    actions = torch.randperm(len(self.tables)).to(self.device)
                else:
                    actions = torch.argsort(self.policy_network(torch.cat((res_details, state_details))), descending=True).to(self.device)

                # Take the action
                action, reward = self.env.step(actions.tolist(), reservation)

                # extra things to record in logging
                total_reward += reward
                actions_taken[action+1] += 1
                if action == len(self.tables):
                    training_rejections += 1
                trajectories[day].append(reward)

                print(f'{print_string}\treward - {reward}')

                # If term state
                if step < len(bookings_on_day)-1:
                    next_res = bookings_on_day.iloc[step+1][self.features]
                else:
                    next_res = None

                # Add to replay buffer
                self.memory.push([current_state, action, reward, self.env.state.detach(), step == len(bookings_on_day)-1, next_res, reservation[self.features]])

                step += 1

                # Epsilon decay
                self.epsilon = max(0.1, self.epsilon_decay*self.epsilon)

                if len(self.memory) < self.batch_size:
                    continue

                # replay sampling
                self.update_networks()
                
            print(f"Day {day}\tproportional_reward={total_reward/len(bookings_on_day):<18}\trejections = {training_rejections}")

        print()
        print(actions_taken)
        print()

        torch.save(self.policy_network.state_dict(), f'{os.getcwd()}/models/DQN-R-{self.restaurant_name}.pt')
        with open(f'{os.getcwd()}/models/DQN-R-{self.restaurant_name}.pkl', 'wb') as f:
            pickle.dump(trajectories, f)

# test_data = pd.read_csv(f'{os.getcwd()}/src/SQL-DATA/Restaurant-{restaurant_name}-test.csv')
# test_data['BookingStartTime'] = (test_data['BookingStartTime'] - 36000) / (60*15)
# test_data['Duration'] = test_data['Duration'] / (60*15)
# test_data["EndTime"] = test_data["BookingStartTime"] + test_data["Duration"]

# print()
# test_predictor(f'Restaurant-{restaurant_name}/DQN', test_data, tables, policy_network, find_table, device, features)