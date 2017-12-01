import pylab as pb
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random

def plotNormal(mean, coVar, xLabel, yLabel, titel):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(titel)
	ax.set_xlabel(xLabel)
	ax.set_ylabel(yLabel)

	x, y = np.mgrid[-4:5:.01, -4:5:0.01]
	pos = np.empty(x.shape + (2,))
	pos[:, :, 0] = x; pos[:, :, 1] = y
	rv = multivariate_normal(mean, coVar)
	
	plt.contourf(x, y, rv.pdf(pos))
	plt.show()

muW = [1.5,-0.8]
covarW = [[1,0],[0,1]]
#plotNormal(muW,covarW, "W0", "W1", "Prio over W")

x = np.arange(-2,2,.02)

xs = np.array([[random.choice(x), 1]])

sigma = np.dot(xs.T,xs)+covarW

t = np.dot(xs, muW)

mu = np.dot(np.linalg.inv(sigma), (np.dot(xs.T, t) + muW))

#print sigma
#print mu

plotNormal(mu,np.linalg.inv(sigma), "W0", "W1", "Posterior over W")

