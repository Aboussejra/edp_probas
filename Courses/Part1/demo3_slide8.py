# Démo n°3, cf. slide 8, Probabilités et EDPs 2021-22
# Sur le mouvement brownien 1D et 2D
# Voir aussi exercice 1, TP n°1
# Attention à bien exécuter une seule cellule à chaque fois
#%%
import numpy as np  # Fonctions de base
from matplotlib import pyplot as plt  # Graphiques
import scipy.stats as stats

# fonction pour simuler un mouvement brownien standard qui utilise une
# boucle for et qui consiste simplement à ajouter des pas indépendants
# A EVITER


def rmbfor(T, N):
    "Simulation d'un mouvement brownien standard"
    path = np.zeros((N + 1))
    for k in range(0, N):
        path[k + 1] = path[k] + np.sqrt(T / N) * np.random.normal(size=1)
    return path


# fonction pour simuler un mouvement brownien standard
# sans utiliser de boucle "for" mais la fonction python np.cumsum
# qui renvoie directement les sommes cumulées d'un vecteur


def rmb(T, N):
    "Simulation d'un mouvement brownien standard"
    path = np.zeros((N + 1))
    pas = np.random.normal(size=N)
    path[1:] = np.sqrt(T / N) * np.cumsum(pas)
    return path


#%%
############################################################################
# Simulation d'un mouvement brownien issu de x, de coefficient de diffusion
# sigma, et sur une durée T avec N pas de temps
############################################################################

# Initialisation
T = 1
N = 1000
x = 0
sigma = 1

NbSimu = 3  # nombre de trajectoires simulées
mb = np.zeros((N + 1, NbSimu))

for k in range(0, NbSimu):
    traj = rmb(T, N)
    mb[:, k] = x + sigma * traj

temps = T * np.linspace(start=0, stop=1, num=N + 1)
plt.figure(facecolor="w")
plt.plot(temps, mb)
plt.plot(temps, 2 * np.sqrt(temps), "k--")
plt.plot(temps, -2 * np.sqrt(temps), "k--")
plt.title("Simulation(s) d" "un mouvement brownien")
plt.xlabel("Temps")
plt.ylabel("Position")
plt.show()

#%%

# Simulations d'un mouvement brownien 1D avec zoom sur la première moitié
# des trajectoires (propriété fractale d'auto-similarité)

plt.close()

plt.figure(facecolor="w")
plt.subplot(211)
plt.plot(temps, mb)
plt.plot(temps, 2 * np.sqrt(temps), "k--")
plt.plot(temps, -2 * np.sqrt(temps), "k--")
plt.title("Zoom (fractalité du MB)")
plt.ylabel("Position")
plt.axis([0, 1, -2.5, 2.5])
plt.subplot(212)
tempsz = 2 * temps[0 : (N // 2 + 1)]
mbz = np.sqrt(2) * mb[0 : (N // 2 + 1)]
plt.plot(tempsz, mbz)
plt.plot(tempsz, 2 * np.sqrt(tempsz), "k--")
plt.plot(tempsz, -2 * np.sqrt(tempsz), "k--")
plt.xlabel("Temps")
plt.ylabel("Position")
plt.axis([0, 1, -2.5, 2.5])
plt.show()

#%%
############################################################################
# Animation brownien 1D
############################################################################

plt.close()

T = 1
N = 500
x = 0
sigma = 1

mb = x + sigma * rmb(T, N)

temps = T * np.linspace(start=0, stop=1, num=N + 1)
plt.figure(facecolor="w")
plt.plot(temps, mb, linestyle="--", color="w")
plt.plot(temps, 2 * np.sqrt(temps), "k--")
plt.plot(temps, -2 * np.sqrt(temps), "k--")
plt.title("Simulation d" "un mouvement brownien")
plt.xlabel("Temps")
plt.ylabel("Position")
plt.axvline(x=0, linestyle="--", color="k")
plt.axhline(y=0, linestyle="--", color="k")
plt.axis([-0.05, T, -2.5 * np.sqrt(T), 2.5 * np.sqrt(T)])
plt.show()

absx = 0
ordy = 0
(line,) = plt.plot(absx, ordy, "ko", ms=4)

for i in range(N):
    ordy = mb[i + 1]
    line.set_ydata(ordy)
    plt.plot(temps[i + 1], mb[i + 1], "o", ms=2, color="red")
    plt.pause(0.01)  # pause en secondes

plt.plot(temps, mb, "r")

#%%
############################################################################
# Simulation du mouvement d'une particule brownienne dans le plan
############################################################################

plt.close()

T = 1
N = 2000
x = (0, 0)
sigma = 1

x1mb = x[0] + sigma * rmb(T, N)
x2mb = x[1] + sigma * rmb(T, N)
plt.figure(facecolor="w")
plt.title("Mouvement brownien plan")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x[0], x[1], "ko", ms=4)
plt.plot(x1mb, x2mb)
plt.plot(x1mb[N], x2mb[N], "ro", ms=4)
plt.axvline(x=x[0], linestyle="dashed", color="r")
plt.axhline(y=x[0], linestyle="dashed", color="r")

plt.gca().set_aspect("equal", adjustable="box")
plt.draw()

#%%
# Simulation 2D (animation)

plt.close()

T = 0.5
sqrT = np.sqrt(T)
plt.figure(facecolor="w")
plt.grid()
absx = 0
ordy = 0
(line,) = plt.plot(absx, ordy, "ko", ms=4)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Mouvement brownien standard")
plt.axis([-2 * sqrT, 2 * sqrT, -2 * sqrT, 2 * sqrT])
plt.axhline(y=0, color="k", linestyle="--")
plt.axvline(x=0, color="k", linestyle="--")
plt.gca().set_aspect("equal", adjustable="box")
plt.draw()

dt = T / 1000
sqrdt = np.sqrt(dt)
sigma = 1
t = 0
for i in range(1000):
    absx_old = absx
    ordy_old = ordy
    absx = absx + sigma * sqrdt * np.random.randn(1)
    ordy = ordy + sigma * sqrdt * np.random.randn(1)
    line.set_xdata(absx)
    line.set_ydata(ordy)
    plt.plot([absx_old, absx], [ordy_old, ordy], linestyle="-", color="blue")
    plt.pause(0.01)  # pause en secondes

plt.plot(absx, ordy, "ro", ms=4)
#%%
############################################################################
# Trajectoire 2D + trajectoires 1D
############################################################################

plt.close()

T = 0.25
N = 500
temps = T * np.linspace(start=0, stop=1, num=N + 1)
sqrT = np.sqrt(T)

plt.figure(facecolor="w")
plt.show()
plt.subplot(311)
absx = 0
ordy = 0
(line,) = plt.plot(absx, ordy, "ko", ms=4)
plt.title("Mouvement brownien 2D")
plt.axis([-2 * sqrT, 2 * sqrT, -2 * sqrT, 2 * sqrT])
plt.axhline(y=0, color="k", linestyle="--")
plt.axvline(x=0, color="k", linestyle="--")
plt.gca().set_aspect("equal", adjustable="box")
plt.draw()

plt.subplot(312)
plt.plot(temps[0], 0, "ko", ms=4)
plt.ylabel("Coordonnée x")
plt.axvline(x=0, linestyle="--", color="k")
plt.axhline(y=0, linestyle="--", color="k")
plt.axis([-0.05, T, -2.5 * np.sqrt(T), 2.5 * np.sqrt(T)])

plt.subplot(313)
plt.plot(temps[0], 0, "ko", ms=4)
plt.xlabel("Temps")
plt.ylabel("Coordonnée y")
plt.axvline(x=0, linestyle="--", color="k")
plt.axhline(y=0, linestyle="--", color="k")
plt.axis([-0.05, T, -2.5 * np.sqrt(T), 2.5 * np.sqrt(T)])


dt = T / N
sqrdt = np.sqrt(dt)
sigma = 1
t = 0
for i in range(N):
    absx_old = absx
    ordy_old = ordy
    absx = absx + sigma * sqrdt * np.random.randn(1)
    ordy = ordy + sigma * sqrdt * np.random.randn(1)
    line.set_xdata(absx)
    line.set_ydata(ordy)
    plt.subplot(311)
    plt.plot([absx_old, absx], [ordy_old, ordy], linestyle="-", color="blue")
    plt.subplot(312)
    plt.plot(temps[i + 1], absx, "bo", ms=1)
    plt.subplot(313)
    plt.plot(temps[i + 1], ordy, "bo", ms=1)
    plt.pause(0.01)  # pause en secondes


###FIN######################################################################
