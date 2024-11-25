import random
import matplotlib.pyplot as plt


#GENERATE RANDOM POINTS

#RED
red_x = []
red_y = []

for i in range(0,20):
    red_x.append(random.randrange(0,19)/4)
    red_y.append(random.randrange(0,19)/4)
for i in range(0,20):
    red_x.append(random.randrange(21,40)/4)
    red_y.append(random.randrange(21,40)/4)

#BLUE
blue_x = []
blue_y = []

for i in range(0,20):
    blue_x.append(random.randrange(21,40)/4)
    blue_y.append(random.randrange(0,19)/4)
for i in range(0,20):
    blue_x.append(random.randrange(0,19)/4)
    blue_y.append(random.randrange(21,40)/4)

print(f'{red_x=}')
print(f'{red_y=}')
print(f'{blue_x=}')
print(f'{blue_y=}')
