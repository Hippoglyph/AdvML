import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

def getposterior(aN,bN,muN,lN,mu,t):
	qmu = ss.norm(muN,1.0/lN).pdf(mu)
	qt = ss.gamma(aN,loc=0,scale=1.0/bN).pdf(t)
	return qmu*qt

def getTruePosterior(a0,b0,mu0,l0,mu,t,N,X):
	value = 1.0
	for n in range(N):
		value *= ss.norm(mu, 1.0/t).pdf(X[n])
	value *= ss.norm(mu0,1.0/(l0*t)).pdf(mu)
	value *= ss.gamma(a0,loc=0,scale=1.0/b0).pdf(t)
	return value
	'''
	newMu = (mu0*l0 + np.sum(X)/(l0+N))
	var = 1/((l0+N)*t)
	qmu = ss.norm(newMu, var).pdf(mu)
	newB = b0+0.5*(l0*mu0**2+np.sum(X**2))
	return qmu*ss.gamma(a0+0.5*N, loc=0, scale = 1/newB).pdf(t)
	'''

def plotDist(aN,bN,muN,lN,a0,b0,mu0,l0,N,X,xLabel, yLabel, titel, res):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(titel)
	ax.set_xlabel(xLabel)
	ax.set_ylabel(yLabel)
	mu = np.linspace(-1,1,res)
	t = np.linspace(0.01,2,res)
	[x,y] = np.meshgrid(mu,t)

	plt.contour(x, y, [[getposterior(aN,bN,muN,lN, m_, t_) for m_ in mu] for t_ in t], colors='b')
	plt.contour(x, y, [[getTruePosterior(a0,b0,mu0,l0, m_, t_, N, X) for m_ in mu] for t_ in t], colors='g')
	plt.show()

N = 10
#X = np.linspace(-1,1,N)
X = np.random.rand(N)
X = (X - np.mean(X))
X /=np.sqrt(np.var(X))
X_ = np.sum(X)/N
X2 = np.sum([xn**2 for xn in X])
iterations = 8
muN = lN = aN = bN = l0 = mu0 = a0 = b0 = 0.000001
muN = (l0*mu0+N*X_)/(l0+N)
aN = a0 + 0.5*(N+1)
for i in range(iterations):
	bN = b0 + 0.5*(X2+l0*mu0**2-2*muN*(N*X_+l0*mu0)+(N+l0)*(muN**2+1.0/lN))
	lN = (l0+N)*(aN/bN)
	#print bN,lN
	#if i == 0:
		#plotDist(aN,bN,muN,lN,a0,b0,mu0,l0,N,X,"mu", "tau", "Estimated posterior", 10)

plotDist(aN,bN,muN,lN,a0,b0,mu0,l0,N,X,"mu", "tau", "Estimated posterior", 50)