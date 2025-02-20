import os
import pandas as pd
import csv

def run(restaurant_name, method_name):

    file_path = f'C:/git/UoB.Y4.Dissertation/src/outputs/Restaurant-{restaurant_name}/{method_name}/'

    all_result_files = os.listdir(file_path)
    if 'results.csv' in all_result_files:
        all_result_files.remove('results.csv')

    results = []

    for file in all_result_files:
        file_name = file.strip('.csv')
        with open(file_path+file, 'r', newline='') as csv_reader:
            reader = csv.reader(csv_reader)
            header = next(reader)

            li = [file_name]
        
            [li.append(int(header[i])) for i in (1,3,5)]
            
            results.append(li)

    df = pd.DataFrame(results, columns=['Date', 'ReservationCount', 'Rejections', 'WastedSlots'])

    df.to_csv(file_path+'results.csv')

    wasted_count = df['WastedSlots'].sum()
    rejections = df['Rejections'].sum()

    print(f'{method_name=}\t{wasted_count=}\t{rejections=}')

        