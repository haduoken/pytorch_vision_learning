import numpy as np
a = np.ones((2,3))
print(a)

a = np.pad(a,((0,0),(1,1)),mode='constant',
                                constant_values=(2,2))
print(a)