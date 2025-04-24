import pandas as pd
import os
import logging
import sys


class LoggerWriter:
    def write(self, message):
        if message.strip(): 
            logging.info(message.strip())

    def flush(self):
        pass

log_file = f'{os.getcwd()}/latex.log'
open(log_file, "w").close()
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(message)s',
    # datefmt='%Y-%m-%d %H:%M:%S'
)

sys.stdout = LoggerWriter()

algos = ['RF', 'LR', 'MLP', 'DQN', 'PPO', 'Non-ML']


for i in range(1,6):

    print(f'\\section{{Restaurant {i}}}')

    for a in algos:

        print(f'\\subsection{{{a}}}')

        row = 0
        section = 1

        results = pd.read_csv(f'{os.getcwd()}/src/results/Restaurant-{i}/{i}-{a}.csv')
        print('\\begin{table}[!ht]')
        print('\\t\\centering')
        print('\\t\\begin{tabular}{c|c|c|c}')
        print('\\tDate & Number of Reservations & Rejections & Wasted Count\\\\')
        row += 1
        print('\\t\\t\\hline')
        for _, result in results.iterrows():
            print(f'\\t\\t{result["Date"]} & {result["ReservationCount"]} & {result["Rejections"]} & {result["WastedCount"]}\\\\')
            row += 1
            if row == 30:
                print('\\t\\end{tabular}')
                print(f'\\t\\caption{{Raw results for {a} in Restaurant {i} (Part {section})}}')
                print(f'\\t\\label{{tab:{a}-{i}}}')
                print('\\end{table}')
                section += 1

                print('\\begin{table}[!ht]')
                print('\\t\\centering')
                print('\\t\\begin{tabular}{c|c|c|c}')
                print('\\tDate & Number of Reservations & Rejections & Wasted Count\\\\')
                print('\\t\\t\\hline')
                row = 1
        print('\\t\\end{tabular}')
        print(f'\\t\\caption{{Raw results for {a} in Restaurant {i} (Part {section})}}')
        print(f'\\t\\label{{tab:{a}-{i}}}')
        print('\\end{table}')
        print('\\n')
        print('\\clearpage')