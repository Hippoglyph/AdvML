import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cdist

def kernel(x1,x2,length):
	return np.exp(-np.dot((x1-x2).T,x1-x2)/length**2)

n = 30
X = np.linspace(-5,5,n).reshape(-1,1)

covar = [[kernel(x1,x2,10) for x1 in X] for x2 in X]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Prior l = 10")
ax.set_xlabel("x")
ax.set_ylabel("t")
for asd in range(3):
	prior = np.random.multivariate_normal([0 for x in range(n)],covar)
	plt.plot(X,prior)

plt.show()