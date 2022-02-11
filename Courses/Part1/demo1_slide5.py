# Démo n°1, cf. slide 5, Probabilités et EDPs 2021-22
# Equation de la chaleur sur une barre "infinie"
# Voir aussi problème 1, TP n°1

import numpy as np
from matplotlib import pyplot as plt

# Données du problème

theta1 = 300  # en Kelvin (K)
theta2 = 500
mu1 = -0.4  # en mètre (m)
mu2 = 0.2
sig1 = 0.2  # en mètre
sig2 = 0.1

sigma = 0.05  # coefficient de diffusion en mètre par (seconde puissance 1/2)

T = 20  # instant final (s)

# Discrétisation de la barre pour le calcul de la température

sigaux = np.maximum(
    np.sqrt(sig1 ** 2 + sigma ** 2 * T), np.sqrt(sig2 ** 2 + sigma ** 2 * T)
)
xlim = 3 * sigaux
pas = xlim / 500

if mu1 < mu2:
    xx = np.arange(start=(mu1 - xlim), stop=(mu2 + xlim), step=pas)
else:
    xx = np.arange(start=(mu2 - xlim), stop=(mu1 + xlim), step=pas)

# condition initiale t = 0

sig1sq = sig1 ** 2
sig2sq = sig2 ** 2

temp0 = theta1 * np.exp(-((xx - mu1) ** 2) / 2 / sig1sq) + theta2 * np.exp(
    -((xx - mu2) ** 2) / 2 / sig2sq
)

# Figure
plt.figure(facecolor="w")


plt.ion()
plt.plot(xx, temp0, linestyle="--", color="k", label="Temps t = 0", linewidth=2)
plt.xlabel("Abscisse x de la barre (m)")
plt.ylabel("Température (K)")
plt.title("Diffusion de la chaleur dans une barre")
plt.axis([1.5 * np.min(xx), 1.5 * np.max(xx), 0, 1.1 * np.max([theta1, theta2])])
plt.grid()
plt.legend()
plt.show()


plt.pause(2)

# Animation
t = 0
for i in range(5 * T + 1):
    t = t + 0.2
    tempT = sig1 * theta1 * np.exp(
        -((xx - mu1) ** 2) / 2 / (sig1sq + t * sigma ** 2)
    ) / np.sqrt(sig1sq + t * sigma ** 2) + sig2 * theta2 * np.exp(
        -((xx - mu2) ** 2) / 2 / (sig2sq + t * sigma ** 2)
    ) / np.sqrt(
        sig2sq + t * sigma ** 2
    )
    if i == 0:
        (line,) = plt.plot(
            xx, tempT, color="r", linewidth=2, label="temps final T = 20 (s)"
        )
    else:
        line.set_ydata(tempT)
    plt.show()
    # plt.pause(0.2)  # pause avec duree en secondes

plt.legend()
