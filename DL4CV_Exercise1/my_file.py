import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = [2, 1, 2]
print(a)
a[:,[0,2]] = 10
a[np.arange(a.shape[0]),1]=0
print(a)
#print(np.logspace(-1, -0.3, 15))

c = np.random.binomial([np.ones((5,4))],1)[0]
print(c)
print(np.random.choice(np.arange(4),size=2))