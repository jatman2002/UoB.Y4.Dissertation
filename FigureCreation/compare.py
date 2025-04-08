import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_wasted(df, name, color):
    new_df = df.sort_values(by='ReservationCount')
    z = np.polyfit(new_df['ReservationCount'],new_df['WastedCount'],1)
    p = np.poly1d(z)

    plt.plot(np.arange(0,80), p(np.arange(0,80)), label=name, color=color)
    plt.xlabel('Daily Reservations')
    plt.ylabel('Wasted Table Frequency')

def plot_rejections(df, name, color):
    new_df = df.sort_values(by='ReservationCount')
    z = np.polyfit(new_df['ReservationCount'],new_df['Rejections'],1)
    p = np.poly1d(z)

    plt.plot(np.arange(0,80), p(np.arange(0,80)), label=name, color=color)
    plt.xlabel('Daily Reservations')
    plt.ylabel('Rejections')

restaurant_name = 1
file_path = f'{os.getcwd()}/src/results/Restaurant-{restaurant_name}/{restaurant_name}-'

existing = pd.read_csv(f'{file_path}Existing-Val.csv')
mlp1 = pd.read_csv(f'{file_path}MLP1.csv')
mlp2 = pd.read_csv(f'{file_path}MLP2.csv')
mlp3 = pd.read_csv(f'{file_path}MLP3.csv')
mlp4 = pd.read_csv(f'{file_path}MLP4.csv')

plt.figure(figsize=(18,6))

plt.subplot(1,3,1)
plt.title('Model Table Utilisation Performance')
plot_wasted(existing, 'existing', 'black')
plot_wasted(mlp1, 'mlp1', 'red')
plot_wasted(mlp2, 'mlp2', 'blue')
plot_wasted(mlp3, 'mlp3', 'green')
plot_wasted(mlp4, 'mlp4', 'orange')
plt.xlim(left=0)
plt.legend()

plt.subplot(1,3,2)
plt.title('Model Reservation Acceptance Performance')
plot_rejections(mlp1, 'mlp1', 'red')
plot_rejections(mlp2, 'mlp2', 'blue')
plot_rejections(mlp3, 'mlp3', 'green')
plot_rejections(mlp4, 'mlp4', 'orange')
plt.xlim(left=0)
plt.legend()

plt.subplot(1,3,3)
ax_1 = existing['ReservationCount'].plot(kind='kde')
ax_1.set_title('Reservation KDE')
ax_1.set_xlabel('Daily Reservations')
ax_1.set_ylabel('Probability')
ax_1.set_xlim(left=0)

# ax_2 = ax_1.twinx()
# mean = existing['ReservationCount'].mean()
# std = existing['ReservationCount'].std()
# x = np.linspace(existing['ReservationCount'].min(), existing['ReservationCount'].max(), 500)
# p = stats.norm.pdf(x, mean, std)
# ax_2.plot(x, p, color='red')

file_p = f'{os.getcwd()}/Img/'
Path(f'{file_p}').mkdir(parents=True, exist_ok=True)

plt.savefig(f'{file_p}MLP-comparison-.pdf', bbox_inches='tight')