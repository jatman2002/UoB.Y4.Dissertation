import random
import matplotlib.pyplot as plt


#RED
red_x = []
red_y = []

for i in range(0,10):
    red_x.append(random.randrange(2,8)/2)
    red_y.append(random.randrange(2,8)/2)
for i in range(0,10):
    red_x.append(random.randrange(10,18)/2)
    red_y.append(random.randrange(10,18)/2)

#BLUE
blue_x = []
blue_y = []

for i in range(0,10):
    blue_x.append(random.randrange(10,18)/2)
    blue_y.append(random.randrange(2,8)/2)
for i in range(0,10):
    blue_x.append(random.randrange(2,8)/2)
    blue_y.append(random.randrange(10,18)/2)


f = plt.figure()
plt.scatter(red_x, red_y, color='#FF0000')
plt.scatter(blue_x, blue_y, color='#0000FF')
plt.show()