import numpy as np
import matplotlib.pyplot as plt


x = np.array([1, 3, 3, 2, 1])
y = np.array([1, 1, 2, 2, 1])


plt.plot(x, y, color='green', linewidth=2, linestyle='-', marker='o', markersize=7)


plt.xlabel('x os')
plt.ylabel('y os')
plt.title('Primjer')


plt.axis([0, 4, 0, 4])
plt.show()