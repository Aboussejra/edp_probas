# Démo n°4, cf. slide 11, Probabilités et EDPs 2020-2021
# Diffusion de grains de chaleur
# Condition initiale = fonction porte (= indicatrice d'un intervalle)

import numpy as np # Fonctions de base
from matplotlib import pyplot as plt # Graphiques
import scipy.stats as stats

# Simulation d'un MB standard

def rmb(T, N):
   "Simulation d'un mouvement brownien standard"   
   path = np.zeros((N+1))
   pas = np.random.normal(size=N)
   path[1:] = np.sqrt(T/N)*np.cumsum(pas)
   return path
   
def indic(x):
    n = len(x)
    y = np.zeros([n,1])
    for i in range(n):
        if x[i] <= b and x[i] >= a:
            y[i] = 1
    else:
            y[i] = 0
    return y

# Condition initaile : indicatrice de l'intervalle [a, b]
a = -0.5
b = 0.5          
xx  = np.linspace(a-3,b+3,500)
yy = indic(xx)

# Initialisation

T = 1
N = 100

p = 10 # nombre de grains

x = np.random.random_sample(p) - 0.5 # [0, -1, 1]
sigma = 1

NbSimu = p
mb = np.zeros((N+1, NbSimu))

for k in range(0, NbSimu):
    traj = rmb(T, N)
    mb[:,k] = x[k] + sigma * traj

temps = T*np.linspace(start=0, stop=1, num=N+1)

plt.figure(facecolor='w')
plt.grid()
plt.plot(temps,mb,linestyle='--',color='w')
plt.title('Diffusion de grains de chaleur')
plt.xlabel('Temps')
plt.ylabel('Position')
plt.axvline(x=0, linestyle='--',color='k')
plt.axhline(y=0, linestyle='--',color='k')
plt.axvline(x=T, linestyle='--',color='k')
plt.axis([-0.3, T+0.3, -3.5*np.sqrt(T), 3.5*np.sqrt(T)])

plt.plot(-0.2*yy, xx, linewidth=2, color='k')      
qb = (b - xx)/np.sqrt(T)
qa = (a - xx)/np.sqrt(T)
prob = stats.norm.cdf(qb,loc= 0,scale= 1) - stats.norm.cdf(qa,loc= 0,scale= 1)
plt.plot(T+0.2*prob, xx, linewidth=2, color='k')
plt.show()

TT = T + np.zeros(p)
x = (b - a)*np.random.random_sample(p) + a # loi uniforme sur [a, b]
ordy = x
plt.plot(np.zeros(p),ordy,'ko', ms=2)
line, = plt.plot(TT,ordy,'ko', ms=2)

for i in range(N):
    ordy = mb[i+1,:]
    line.set_ydata(ordy)
    for k in range(p):
        plt.plot(temps[i+1],mb[i+1,k],'ko', ms=2, color='r')
    plt.pause(0.0001) # pause en secondes

plt.show()