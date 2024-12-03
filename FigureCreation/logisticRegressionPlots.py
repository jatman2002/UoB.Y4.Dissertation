from matplotlib import pyplot as plt
import math
import numpy as np
 
x = np.arange(-10,10, 0.1)
y = 1 / (1 + np.exp(x*-1))

plt.gcf().set_size_inches(6,4)
plt.plot(x,y)
plt.savefig("Img/LogisticRegression/LogisticFunction.pdf")

