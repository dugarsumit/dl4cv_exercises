import numpy as np
from collections import Counter

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = [2, 1, 2]
a = Counter({'a':1,'b':2})
b = Counter({'a':1,'b':2})
c = {'d':1,'s':5}
print(c)
c = dict(a+b)
c.update((k,v/2) for k,v in c.items())
print(np.logspace(0.1, 0.2, 15))
print(1e-7)