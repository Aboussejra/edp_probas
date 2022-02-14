# TP n°1, EDPs et Probabilités 2021-22
# Problème 1 - Résolution de l'équation de la chaleur par méthode Monte-Carlo
# sur une barre "infinie"
#%%
from cProfile import label
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
    np.sqrt(sig1**2 + sigma**2 * T), np.sqrt(sig2**2 + sigma**2 * T)
)
xlim = 3 * sigaux
pas = xlim / 500
if mu1 < mu2:
    xx = np.arange(start=(mu1 - xlim), stop=(mu2 + xlim), step=pas)
else:
    xx = np.arange(start=(mu2 - xlim), stop=(mu1 + xlim), step=pas)

# Calcul de la température initiale (t = 0)

sig1sq = sig1**2
temp1 = np.exp(-((xx - mu1) ** 2) / 2 / sig1sq)
sig2sq = sig2**2
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

sig1sqT = sig1sq + T * sigma**2
temp1 = np.exp(-((xx - mu1) ** 2) / 2 / sig1sqT)
sig2sqT = sig2sq + T * sigma**2
temp2 = np.exp(-((xx - mu2) ** 2) / 2 / sig2sqT)
tempT = sig1 * theta1 * temp1 / np.sqrt(sig1sqT)
tempT += sig2 * theta2 * temp2 / np.sqrt(sig2sqT)
plt.plot(xx, tempT, label="Temps T")
plt.legend(loc="best")

#%%
# Générateur de Brownien
def Brownien(position, T):
    position = position + sigma * np.sqrt(T) * np.random.normal(size=1)
    return position


def theta(x, t, NMC):
    x_MC = []
    for i in range(NMC):
        x_MC.append(Brownien(x, t))
    x_MC = np.array(x_MC)
    temp1_MC = np.exp(-((x_MC - mu1) ** 2) / 2 / sig1sq)
    temp2_MC = np.exp(-((x_MC - mu2) ** 2) / 2 / sig2sq)
    tempT_MC = theta1 * temp1_MC + theta2 * temp2_MC

    THat = np.mean(tempT_MC)
    se = np.std(tempT_MC) / np.sqrt(NMC)
    return THat, se


x = xx[600]
NMC = 40
t = T
THat, se = theta(x, t, NMC)
print(
    "Theta({:.2f},{:.2f}) IC à 95% : [{:.2f};{:.2f}]".format(
        x, t, THat - 2 * se, THat + 2 * se
    )
)
print(f"true one is {tempT[600]}")
theta_MC_list = []
theta_minus_2_se = []
theta_plus_2_se = []
for x in xx:
    THat, se = theta(x, t, NMC)
    theta_MC_list.append(THat)
    theta_minus_2_se.append(THat - 2 * se)
    theta_plus_2_se.append(THat + 2 * se)
plt.figure(facecolor="w")
plt.plot(xx, theta_MC_list, label="simulation MC = 400")
# plt.plot(xx, theta_minus_2_se, label="theta - 2*se")
# plt.plot(xx, theta_plus_2_se, label="theta + 2*se")
plt.plot(xx, tempT, label="Temps T")
plt.legend(loc="best")
# Produire graph T = 0, T = T, estimation monte carlo plot. Erreur plot intervals de confiance
# %%
# Partie 2 : méthode particulaire
# Estimation densité scipy

# Simulation de particules :

# Données du problème :
from scipy import stats

theta1 = 300  # en Kelvin (K)
theta2 = 500
mu1 = -0.4  # en mètre (m)
mu2 = 0.2
sig1 = 0.2  # en mètre
sig2 = 0.1
ratio = theta1 * sig1 / (theta1 * sig1 + theta2 * sig2)
normalisation = np.sqrt(2 * np.pi) * (sig1 * theta1 + sig2 * theta2)

sigma = 0.05  # coefficient de diffusion en mètre par seconde puissance 1/2

# We want to generate a particle from a bigaussian.
# We simulate a bernouilli law of parameter ratio to simulate the chance of picking from which gaussian an pick from it


def generate_particle():
    choice = np.random.binomial(size=1, n=1, p=ratio)
    if choice == 1:
        particle = np.random.normal(mu1, sig1, size=1)
    else:
        particle = np.random.normal(mu2, sig2, size=1)
    return particle


t = T
NMC = 400
pop_list = []
for i in range(NMC):
    pop = generate_particle()
    pop += Brownien(pop, t)
    pop_list.append(pop)
pop_list = np.concatenate(pop_list)
density_estimation = stats.gaussian_kde(pop_list)
plt.figure(facecolor="w")
plt.plot(xx, tempT, label="Temps T")
plt.xlabel("Abscisse x de la barre (m)")
plt.ylabel("Température (K)")
plt.plot(xx, density_estimation(xx) * normalisation, label="Estimation MC")
plt.grid()
plt.legend(loc="best")

# %%
# Problème 2 Condition de Dirichlet

a = 1
b = 1
theta_a = 100
theta_b = 500
sigma = 0.01
n1 = 5
n2 = 8
theta1 = 50
theta2 = 100
t = 60
x = 0
NMC = 400
N = 100


def theta_inf(x):
    return theta_a + (theta_b - theta_a) * (x + a) / (b + a)


def theta_zero(x):
    return (
        theta_inf(x)
        + theta1 * np.sin(2 * np.pi * n1 * (x + a) / (2 * (b + a)))
        + theta2 * np.sin(2 * np.pi * n2 * (x + a) / (2 * (b + a)))
    )


def theta_an(x, t):
    return (
        theta_inf(x)
        + theta1
        * np.exp(-pow(np.pi * n1 * sigma / (b + a), 2) * t / 2)
        * np.sin(2 * np.pi * n1 * (x + a) / (2 * (b + a)))
        + theta2
        * np.exp(-pow(np.pi * n2 * sigma / (b + a), 2) * t / 2)
        * np.sin(2 * np.pi * n2 * (x + a) / (2 * (b + a)))
    )


def theta_MC(x, t, NMC):
    theta_estimates = []
    for i in range(NMC):
        delta_t = t / N
        xc = x
        s = 0
        while x > -a and x < b and s < t:
            xc = xc + sigma * np.sqrt(delta_t) * np.random.normal(size=1)
            s = s + delta_t
        if s >= t:
            theta_estimates.append(theta_zero(xc))
        elif xc <= -a:
            theta_estimates.append(theta_a)
        elif xc >= b:
            theta_estimates.append(theta_b)
        else:
            print("problem")
    THat = np.mean(theta_estimates)
    se = np.std(theta_estimates) / np.sqrt(NMC)
    return THat, se


THat, se = theta_MC(x, t, NMC)

print(theta_an(x, t))

print(
    "Theta({:.2f},{:.2f}) IC à 95% : [{:.2f};{:.2f}]".format(
        x, t, THat - 2 * se, THat + 2 * se
    )
)
# %%
