# Application of Machine Learning techniques in the context of restaurant table allocations

## Usage
For training, run the following command:<br>
```
python train.py [-h] [-r RESTAURANT] [-g GPU] [-a ALGO] [-l] [-s START_POINT]
```
Options:
```
-r RESTAURANT     Specify which restaurant  
-g GPU            Specify which GPU to use (optional) 
-a ALGO           Specify which algorithm to use  
-l                Disable logging to log file (enables printing to terminal) (optional)
-s START_POINT    Start point for grid search for MLP (optional)
```

For model testing on testing set, run the following command:<br>
```
python src/ModelTesting/run.py [-h] [-r RESTAURANT] [-a ALGO] [-v] [-R]
```
Options:
```
-r RESTAURANT  specify which restaurant
-a ALGO        specify which algorithm to use
-v             use validation set (optional) 
-R             Collalte results (optional) 
```
Inference was done on CPU so specifying GPU was not needed, in the case of Keras models, the visible devices was set to -1 (i.e. no visible GPUs) by:
```
CUDA_VISIBLE_DEVICES=-1 python src/ModelTesting/run.py [args]
```

The outputs of each model when testing are stored under `src/outputs/Restuarant-[r]/[method-name]` and each file in these directories is named by the date of the set of bookings tested and look similar to the diagram below:
| REQUEST COUNT | value | REJECTED | value | WASTED COUNT | value |
|---------------|-------|----------|-------|--------------|-------|

| | | | | |
|-------|-------|-------|-------|-------|
|       |       | B1    | B1    | B1    |
| B2    | B2    |       |       | B3    |
|       | B4    | B4    | B4    |       |


The collated results for each restaurant and method get stored under  `src/results/Restuarant-[r]/[method-name]` and look like:
| | Date | ReservationCount | Rejections | WastedCount |
|-|------|------------------|------------|-------------|
| |      |                  |            |             |

NOTE: THE DATASET MUST BE STORED EXACTLY AS DESCRIBED BELOW IN THE CSV DATA SECTION OTHERWISE CODE WILL NOT WORK!!!

## Code structure
All the Python scripts can be found under `src/`.<br>
`Algorithms/` contains the code for the Reallocation algorithm.

`ML_methods/` contains all the code for training the Supervised Learning models.<br>
It also contains another director called `helper/`. This contains code to generate the soft labels and state representations and add them to the bookings CSVs. Also contains code to test models on a test or validation set and collate results.

`RL` contains all the code for the Reinforcement Learning methods, including trianing loops and custom environment.<br>
Also contains code to test models on a test or validation set. `ML-methods/helper/Results.py` was used to collate results for RL methods as well.

`ModelTesting/` contains all the code for final testing on the test set for ALL models. `ModelClasses` contain the classes for each ML method used in the dissertation with `Results.py` collating final results for each method.

`dataset-related-code/` contains code to get the data from the SQL server, apply pre-processing, split into train and testing, and gather metrics for the ground truth allocations for later comparison.

## CSV data
The dataset is stored under `src/SQL-DATA/`<br>
The CSV files marked as `Restaurant-[x]-bookings.csv` are reservations (and allocations) given by the pre-processing step discussed in the dissertation <br>
| BookingCode | GuestCount | BookingDate | BookingTime | Duration | CreatedOn | TableCode |
|-------------|------------|-------------|-------------|----------|-----------|-----------|

The ones marked as `Restaurant-[x]-train.csv` are the reservations for each restaurant in their respective training dataset. <br>
The ones marked as `Restaurant-[x]-test.csv` are the reservations for each restaurant in their respective testing dataset.

The CSV files marked as `Restaurant-[x]-tables.csv` contain a list of tables for each restaurant<br>
|SiteCode | TableCode | MinCovers | MaxCovers|
|---------|-----------|-----------|----------|

There is also a zip file called `SQL-DATA.zip`. This zip file contains the above-mentioned files. It additionally contains three other directories:
* MLP-State <br>
  This contains the bookings for each restaurant, including train/test files. These files contain an additional set of columns to represent the state of the restaurant at the time of a booking (refer to the dissertation, Chapter 4, for the specifics)
* MLP-Soft-Encoding <br>
  This contains the bookings for each restaurant, including train/test files. These files contain additional headers, one for each table, and the values denote the probability of a table being the correct one for a given booking (refer to the dissertation, Chapter 4, for the specifics)
* MLP-State-Soft-Label <br>
  This combines additional headers from the above two directories

