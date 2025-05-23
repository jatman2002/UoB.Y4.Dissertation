import os
import pandas as pd
import csv
from pathlib import Path

def run(restaurant_name, method_name, MLP=None):

    file_path = f'/mnt/fast0/jy894/outputs/Restaurant-{restaurant_name}/{method_name}/'
    if not MLP is None:
        file_path = f'{file_path}{MLP}/'

    Path(file_path).mkdir(parents=True, exist_ok=True)

    all_result_files = os.listdir(file_path)
    if 'results.csv' in all_result_files:
        all_result_files.remove('results.csv')

    all_result_files = sorted(all_result_files)

    results = []

    for file in all_result_files:
        file_name = file.strip('.csv')
        with open(file_path+file, 'r', newline='') as csv_reader:
            reader = csv.reader(csv_reader)
            header = next(reader)

            li = [file_name]
        
            [li.append(int(header[i])) for i in (1,3,5)]
            
            results.append(li)

    df = pd.DataFrame(results, columns=['Date', 'ReservationCount', 'Rejections', 'WastedCount'])


    Path(f'/mnt/fast0/jy894/results/Restaurant-{restaurant_name}/').mkdir(parents=True, exist_ok=True)

    results_path = f'/mnt/fast0/jy894/results/Restaurant-{restaurant_name}/{restaurant_name}-{method_name}'
    if not MLP is None:
        results_path = f'{results_path}-{MLP}'

    df.to_csv(f'{results_path}.csv')

    wasted_count = df['WastedCount'].sum()
    rejections = df['Rejections'].sum()

    print(f'{method_name=}\t{wasted_count=}\t{rejections=}')

        