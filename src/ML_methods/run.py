# import LR
import MLPKeras
import MLP
# import OVR
# import RF
# import GridSearch.RFGridSearch as RFGridSearch
# import GridSearch.LRGridSearch as LRGridSearch

import helper.Results as Results

restaurant_name = '2'
# RFGridSearch.run(restaurant_name)
# Results.run(restaurant_name, 'RFGridSearch')
# LRGridSearch.run(restaurant_name)
# Results.run(restaurant_name, 'LRGridSearch')


# LR.run(restaurant_name)
MLPKeras.run(restaurant_name)
MLP.run(restaurant_name)
# OVR.run(restaurant_name)
# RF.run(restaurant_name)
# Results.run(restaurant_name, 'RL-test')
# Results.run(restaurant_name, 'Existing')


methods = ['MLP', 'MLPKeras', 'Existing']
for method in methods:
    Results.run(restaurant_name, method)


# for restaurant_name in range(1,6):
#     Results.run(restaurant_name, 'Existing')
    
#     LR.run(restaurant_name)
#     Results.run(restaurant_name, 'LR')

#     # MLP.run(restaurant_name)
#     # Results.run(restaurant_name, method)

#     OVR.run(restaurant_name)
#     Results.run(restaurant_name, 'OVR')

#     RF.run(restaurant_name)
#     Results.run(restaurant_name, 'RF')
