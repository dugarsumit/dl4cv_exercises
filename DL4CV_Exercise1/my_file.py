import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = [2, 1, 2]
#print(np.hstack((a, b)))
#print(np.append(a,b,axis=0))
print(np.column_stack((a,b)))
print(np.row_stack((a,b)))