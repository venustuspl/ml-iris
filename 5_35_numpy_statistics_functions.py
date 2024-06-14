import numpy as np

data = np.array([[10, 7, 4], [3, 2, 1]])
data

np.median(data)

np.median(data, axis=0)

np.median(data, axis=1)

np.mean(data)
np.mean(data, axis=0)
np.mean(data, axis=1)

np.average(data)
np.average(data, axis=0)
np.average(data, axis=1)

data
np.average(data, weights=[0, 1, 1], axis=1)

np.average(data, weights=[2, 3, 5], axis=1)
# (2*10 + 3*7 + 5*4)/10

np.var(data)
np.var(data, axis=0)
np.var(data, axis=1)

np.std(data)
np.std(data, axis=0)
np.std(data, axis=1)

data = np.zeros((2, 100000))
data[0, :] = 1.0
data[1, :] = 0.1
np.mean(data, dtype=np.float32)
np.mean(data, dtype=np.float64)

data = np.zeros((2, 10))
data[0, :] = 1.0
data[1, :] = 0.1
np.mean(data, dtype=np.float32)
np.mean(data, dtype=np.float64)