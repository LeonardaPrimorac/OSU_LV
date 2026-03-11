import numpy as np
import matplotlib.pyplot as plt


black_square = np.zeros((50, 50))  
white_square = np.ones((50, 50))  


upper_row = np.hstack((black_square, white_square))  
lower_row = np.hstack((white_square, black_square))  
final_image = np.vstack((upper_row, lower_row))      


plt.imshow(final_image, cmap='gray', interpolation='nearest')
plt.title('Četiri kvadrata')
plt.show()