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

	x, y = np.mgrid[-5:5:.01, -5:5:0.01]
	pos = np.empty(x.shape + (2,))
	pos[:, :, 0] = x; pos[:, :, 1] = y
	rv = multivariate_normal(mean, coVar)
	
	plt.contourf(x, y, rv.pdf(pos))
	plt.show()

def getSigma(X,sig, t):
	return np.linalg.inv(np.dot(X.T,X)/sig + np.identity(2)/t)

def getMu(sigma, X,T, muW0, sig, t):
	return np.array(np.dot(sigma, np.dot(X.T,T)/sig + muW0*(1/t))).reshape(-1)


Worig = [1.5,-0.8]

muW0 = np.matrix([[0,0]]).T
covarW0 = [[1,0],[0,1]]
x_ = np.arange(-2,2,.02)
x = [[x__, 1] for x__ in x_]

#plotNormal(muW0,covarW0, "W0", "W1", "Prio over W")
xs = np.matrix([random.choice(x) for i in range(50)])

T = (np.dot(xs, Worig) + np.random.normal(0,0.2, len(xs))).T

sigma = getSigma(xs,1,1)
mu = getMu(sigma, xs, T, muW0, 1, 1)

plotNormal(mu,sigma, "W0", "W1", "Posterior over W")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Functions")
ax.set_xlabel("x")
ax.set_ylabel("t")
for asd in range(3):
	W = np.random.multivariate_normal(mu,sigma)
	print W
	yGuess = np.dot(x,W)
	plt.plot(x_,yGuess.T)


plt.plot(xs[:,0:1], T, "x")
#yTrue = np.dot(x,Worig)
#plt.plot(x_,yTrue.T, "--")
plt.show()