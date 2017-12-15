import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

def getposterior(aN,bN,muN,lN,mu,t):
	qmu = ss.norm(muN,1.0/lN).pdf(mu)
	qt = ss.gamma(aN,loc=0,scale=1.0/bN).pdf(t)
	return qmu*qt

def getTruePosterior(a0,b0,mu0,l0,mu,t,N,X):
	newMu = (mu0*l0 + np.sum(X)/(l0+N))
	var = 1/((l0+N)*t)
	qmu = ss.norm(newMu, var).pdf(mu)
	newB = b0+0.5*(l0*mu0**2+np.sum(X**2))
	return qmu*ss.gamma(a0+0.5*N, loc=0, scale = 1/newB).pdf(t)

def plotDist(aN,bN,muN,lN,a0,b0,mu0,l0,N,X,xLabel, yLabel, titel, res):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(titel)
	ax.set_xlabel(xLabel)
	ax.set_ylabel(yLabel)
	mu = np.linspace(0.8,2.2,res)
	t = np.linspace(0,1,res)
	[x,y] = np.meshgrid(mu,t)

	plt.contour(x, y, [[getposterior(aN,bN,muN,lN, m_, t_) for m_ in mu] for t_ in t], colors='b')
	#plt.contour(x, y, [[getTruePosterior(a0,b0,mu0,l0, m_, t_, N, X) for m_ in mu] for t_ in t], colors='g')
	plt.show()

N = 10
X = np.linspace(1,2,N)
X_ = np.mean(X)
iterations = 5
muN = lN = aN = bN = l0 = mu0 = a0 = b0 = 0.0
lN = 0.5
muN = (l0*mu0+N*X_)/(l0+N)
aN = a0 + N*0.5
for i in range(iterations):
	bN = b0 + 0.5*(np.sum([xn**2 for xn in X])+l0*mu0**2-2*(N*X_+l0*mu0)*muN+(l0+N)*(muN**2+1/lN))
	lN = (l0+N)*(aN/bN)
	print bN,lN
	#if i == 0:
		#plotDist(aN,bN,muN,lN,a0,b0,mu0,l0,N,X,"mu", "tau", "Estimated posterior", 10)

plotDist(aN,bN,muN,lN,a0,b0,mu0,l0,N,X,"mu", "tau", "Estimated posterior", 10)