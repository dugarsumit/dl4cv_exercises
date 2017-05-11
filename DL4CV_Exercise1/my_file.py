import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a[[1,2]])
b = [2,1,2]
print(a)
print(np.arange(3))
print(a[np.arange(3),b])