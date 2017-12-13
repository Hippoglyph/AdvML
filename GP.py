import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.spatial.distance import cdist

def kernel(x1,x2,length):
	return np.exp(-np.dot((x1-x2).T,x1-x2)/length**2)

def getK(x1,x2):
	return [[kernel(x1_,x2_,l) for x1_ in x1] for x2_ in x2]+np.eye(len(x1))*0

def calMu(x,Ct):
	k = [kernel(xi,x,l) for xi in X]
	return np.dot(k,Ct)

def getMu(T, prior, points):
	Ct = np.dot(np.linalg.inv(prior), T)
	return np.array([calMu(xi,Ct) for xi in points]).reshape(-1,1)

def calVar(x,X, C):
	k = np.array([kernel(xi,x,l) for xi in X])
	return 1 - np.dot(np.dot(k,C), k.reshape(-1,1))

def getVar(X,prior, points):
	C = np.linalg.inv(prior)
	return np.sqrt([calVar(xi,X,C) for xi in points])

l = 1
pi = math.pi
n = 7
X = np.linspace(0,2*pi,n).reshape(-1,1)

pCovar = getK(X,X)
#prior = np.random.multivariate_normal([0 for x in range(n)],pCovar, 5)

T = [math.cos(x)+np.random.normal(0,0.5) for x in X]
newX = np.linspace(0,2*pi,100).reshape(-1,1)

mu = getMu(T,pCovar,newX)
var = getVar(X,pCovar, newX)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Posterior l = 1")
ax.set_xlabel("x")
ax.set_ylabel("t")
plt.plot(X,T, "bs")
plt.plot(newX,mu, "r--")
plt.gca().fill_between(newX.flat, (mu-2*var).flat, (mu+2*var).flat, color="#dddddd")
plt.show()




'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Prior l = 1")
ax.set_xlabel("x")
ax.set_ylabel("t")
for p in prior:
	plt.plot(Xprior,p)

plt.show()
'''