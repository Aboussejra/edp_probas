# Démo n°2, cf. slide 7, Probabilités et EDPs 2021-22
# Illustration de la propriété de régularisation (instantanée!) par convolution 
# avec le noyau de la chaleur
# Voir exercice 2.1

from math import *
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

# Données du problème

a = -1
b = 1

# fonction indicatrice de l'intervalle [a,b]
# appelée encore fonction porte, créneau ou fenêtre

def indic(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        if x[i] <= b and x[i] >= a:
            y[i] = 1
        else:
            y[i] = 0
    return y
            
xx  = np.linspace(a-1,b+1,500)
yy = indic(xx)

# Graphe

plt.figure(facecolor='w')
plt.plot(xx, yy, linewidth=2, color='b',label="Fonction indicatrice")
plt.xlabel("x")
plt.ylabel("x")
plt.title("Régularisation par convolution avec le noyau de la chaleur")
plt.axis([a-1, b+1, -0.2, 1.2])
plt.axhline(y=0,color='k')
plt.axvline(x=0,color='k')

# Convolution avec le noyau de la chaleur  

t = 0.1       
qb = (b - xx)/np.sqrt(t)
qa = (a - xx)/np.sqrt(t)

prob = stats.norm.cdf(qb,loc= 0,scale= 1) - stats.norm.cdf(qa,loc= 0,scale= 1)
plt.plot(xx, prob, linewidth=2, color='red', label="Régularisée : t = 0.1")
plt.legend(loc="best")

t = 0.01       
qb = (b - xx)/np.sqrt(t)
qa = (a - xx)/np.sqrt(t)

prob = stats.norm.cdf(qb,loc= 0,scale= 1) - stats.norm.cdf(qa,loc= 0,scale= 1)
plt.plot(xx, prob, linewidth=2, color='green', label= "Régularisée : t = 0.01 ")
plt.legend(loc="best")

t = 0.001       
qb = (b - xx)/np.sqrt(t)
qa = (a - xx)/np.sqrt(t)

prob = stats.norm.cdf(qb,loc= 0,scale= 1) - stats.norm.cdf(qa,loc= 0,scale= 1)
plt.plot(xx, prob, linewidth=2, color='black', label= "Régularisée : t = 0.001 ")
plt.legend(loc="best")
plt.axhline(y=0.5,linestyle='--',color='k')

