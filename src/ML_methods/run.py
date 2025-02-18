import LR
import MLP
import OVR
import RF

import Results

restaurant_name = '1'

LR.run(restaurant_name)
MLP.run(restaurant_name)
OVR.run(restaurant_name)
RF.run(restaurant_name)


methods = ['LR', 'MLP', 'OVR', 'RF']
for method in methods:
    Results.run(restaurant_name, method)