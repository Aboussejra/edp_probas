# TP n°1, EDPs et Probabilités 2021-22
# Problème 1 - Résolution de l'équation de la chaleur par méthode Monte-Carlo
# sur une barre "infinie"
#%%
from math import *
import numpy as np
from matplotlib import pyplot as plt

##############################################################################
# Partie 1 : calcul de la température au point x et au temps T par la méthode
# Monte-Carlo "brute"
##############################################################################

# Données du problème

theta1 = 300  # en Kelvin (K)
theta2 = 500
mu1 = -0.4  # en mètre (m)
mu2 = 0.2
sig1 = 0.2  # en mètre
sig2 = 0.1

sigma = 0.05  # coefficient de diffusion en mètre par seconde puissance 1/2

# Discrétisation de la barre pour le calcul de la température
# à l'instant T

T = 10

sigaux = np.maximum(
    np.sqrt(sig1 ** 2 + sigma ** 2 * T), np.sqrt(sig2 ** 2 + sigma ** 2 * T)
)
xlim = 3 * sigaux
pas = xlim / 500
if mu1 < mu2:
    xx = np.arange(start=(mu1 - xlim), stop=(mu2 + xlim), step=pas)
else:
    xx = np.arange(start=(mu2 - xlim), stop=(mu1 + xlim), step=pas)

# Calcul de la température initiale (t = 0)

sig1sq = sig1 ** 2
temp1 = np.exp(-((xx - mu1) ** 2) / 2 / sig1sq)
sig2sq = sig2 ** 2
temp2 = np.exp(-((xx - mu2) ** 2) / 2 / sig2sq)
temp0 = theta1 * temp1 + theta2 * temp2

# Graphe

plt.figure(facecolor="w")
plt.plot(xx, temp0, label="Temps 0")
plt.xlabel("Abscisse x de la barre (m)")
plt.ylabel("Température (K)")
plt.grid()
plt.title("Diffusion de la chaleur dans une barre")

# Evolution de la température jusqu'au temps T
T = 10  # en seconde (s)

# Température au temps T et graphe

sig1sqT = sig1sq + T * sigma ** 2
temp1 = np.exp(-((xx - mu1) ** 2) / 2 / sig1sqT)
sig2sqT = sig2sq + T * sigma ** 2
temp2 = np.exp(-((xx - mu2) ** 2) / 2 / sig2sqT)
tempT = sig1 * theta1 * temp1 / np.sqrt(sig1sqT)
tempT += sig2 * theta2 * temp2 / np.sqrt(sig2sqT)
print(tempT)
plt.plot(xx, tempT, label="Temps T")
plt.legend(loc="best")

#%%
# Générateur de marche aléatoire sur un réseau (Random Walk)
def RW(position, pas, N):
    for i in range(N):
        direction = np.random.randint(low=1, high=3, size=1)
        if direction == 1:
            step = pas * 1
        else:
            step = pas * -1
        position = position + step
    return position


x = xx[0]
NMC = 400
x_MC = []
T = 10
N = 100
h = T / N
for nMC in range(NMC):
    x_MC.append(RW(x, h, N))

x_MC = np.array(x_MC)
temp1_MC = np.exp(-((x_MC - mu1) ** 2) / 2 / sig1sq)
temp2_MC = np.exp(-((x_MC - mu2) ** 2) / 2 / sig2sq)
tempT_MC = theta1 * temp1_MC + theta2 * temp2_MC

THat = np.mean(tempT_MC)
print(THat)
print(tempT)
plt.plot(xx, tempT)
# %%
