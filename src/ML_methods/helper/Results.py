import os
import pandas as pd
import csv
from pathlib import Path

def run(restaurant_name, method_name):

    file_path = f'{os.getcwd()}/src/outputs/Restaurant-{restaurant_name}/{method_name}/'

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


    Path(f'{os.getcwd()}/src/results/Restaurant-{restaurant_name}/').mkdir(parents=True, exist_ok=True)

    df.to_csv(f'{os.getcwd()}/src/results/Restaurant-{restaurant_name}/{restaurant_name}-{method_name}.csv')

    wasted_count = df['WastedCount'].sum()
    rejections = df['Rejections'].sum()

    print(f'{method_name=}\t{wasted_count=}\t{rejections=}')

        