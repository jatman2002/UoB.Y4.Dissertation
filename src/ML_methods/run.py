import LR
import MLP
import OVR
import RF

import Results

import matplotlib.pyplot as plt

restaurant_name = '1'

# LR.run(restaurant_name)
# MLP.run(restaurant_name)
# OVR.run(restaurant_name)
# RF.run(restaurant_name)


# methods = ['LR', 'MLP', 'OVR', 'RF']
# for method in methods:
#     Results.run(restaurant_name, method)

# Results.run(restaurant_name, 'MLP')

alphas = [0.5]
li = []

for alpha in alphas:
    li.append(MLP.run(restaurant_name, alpha))

f = plt.figure()
for l, a in zip(li, alphas):
    plt.plot([i for i in range(1,101)], l, label=a)
plt.legend()
f.savefig("C:/git/UoB.Y4.Dissertation/src/mlp.pdf")