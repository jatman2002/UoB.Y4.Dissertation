import LR
import MLP
import OVR
import RF
import RFGridSearch

import Results

restaurant_name = '1'
RFGridSearch.run(restaurant_name)
Results.run(restaurant_name, 'RFGridSearch')

# LR.run(restaurant_name)
# MLP.run(restaurant_name)
# OVR.run(restaurant_name)
# RF.run(restaurant_name)
# Results.run(restaurant_name, 'RF')


# methods = ['LR', 'MLP', 'OVR', 'RF']
# for method in methods:
#     Results.run(restaurant_name, method)


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
