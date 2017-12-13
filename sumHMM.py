import random as rnd
import numpy as np
import matplotlib.pyplot as plt
K = 100
N = 100
D = [[2.0/15,2.0/15,2.0/15,2.0/15,2.0/15,1.0/3],
	 [1.0/3,2.0/15,2.0/15,2.0/15,2.0/15,2.0/15]]
V = [1,2,3,4,5,6]

def playTable(prevDiceIdx):
	if rnd.random() < 1.0/3:
		return (prevDiceIdx, np.sum(np.random.choice(V,1, p=D[prevDiceIdx])))
	return ((prevDiceIdx+1)%2, np.sum(np.random.choice(V,1, p=D[(prevDiceIdx+1)%2])))

def play():
	table = playTable(0)
	X = [table[1]]
	for k in range(K-1):
		table = playTable(table[0])
		X.append(table[1])
	return X

acc = [0,0,0,0,0,0]
for n in range(N):
	X = play()
	for x in X:
		acc[x-1]+=1

fig,ax = plt.subplots()
ax.set_ylabel('Occurences')
rec = ax.bar(V,acc,0.35,color = 'r')
plt.show()