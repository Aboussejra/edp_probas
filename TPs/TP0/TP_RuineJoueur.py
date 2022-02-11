# Introduction au cours "EDPs et Probabilités 2021-22"
# Sur la ruine du joueur


import numpy as np                    
import random as rnd                  
from matplotlib import pyplot as plt  

#%% 

# Fonction pour simuler une marche aléatoire standard sur une grille 1D
# et qui consiste simplement à ajouter des pas indépendants
# sans utiliser de boucle "for" mais la fonction python np.cumsum
# qui renvoie directement les sommes cumulées d'un vecteur
# Voir  l'aide sur np.cumsum
 

def Rwalk(N, p):
   "Simulation d'une marche aléatoire sur une grille"   
   path = np.zeros((N+1))
   pas = 2*np.random.binomial(n=1, p=p, size=N) - 1
   path[1:] = np.sqrt(1/N)*np.cumsum(pas)
   return path

# Initialisation
N = 1000

NbSimu = 3 # nombre de trajectoires simulées
mb = np.zeros((N+1, NbSimu))

for k in range(0, NbSimu):
    traj = Rwalk(N, 0.5)
    mb[:,k] = traj

temps = np.linspace(start=0, stop=1, num=N+1)
plt.figure(facecolor='w')
plt.plot(temps,mb)
plt.plot(temps,2*np.sqrt(temps),'k--')
plt.plot(temps,-2*np.sqrt(temps),'k--')
plt.title("Simulation(s) d'un mouvement brownien")
plt.xlabel('Temps')
plt.ylabel('Position')
plt.grid()
plt.show()

#%%
# Calcul de la probabilité de ruine b/(a+b)
a = 9     # fortune du joueur
b = 1     # gain cible 

p_ruine = b/(a+b)   # calcul théorique

# Calcul par méthode Monte-Carlo (estimation Monte-Carlo)
NMC = 4000       # nombre de simulations Monte-Carlo
compteur = 0    # on compte le nombre de fois correspondant à la ruine
temps_moyen = 0 # on en profite pour estimer la durée moyenne du jeu

for k in range(NMC):
    rw = 0  # marche aléatoire qui démarre de 0
    temps = 0
    while (rw > -a) and (rw < b): 
            rw += rnd.choice([-1, 1])
            temps += 1
    if rw == -a: 
        compteur += 1
    temps_moyen += temps # attention à l'indentation!


# Estimation ponctuelle de la probabilité de ruine
p_MC = compteur/NMC
sig_MC = np.sqrt(p_MC*(1-p_MC)/NMC)

print("**************************************************")
print("La proba de ruine estimée est de :")
print("          ",'{:.3f}'.format(p_MC))
print("pour NMC =", NMC, "simulations Monte-Carlo.")
print("\nIntervalle de confiance à 95% :")
print("     [",'{:.3f}'.format(p_MC - 2*sig_MC),";",
        '{:.3f}'.format(p_MC + 2*sig_MC),"]")
print("**************************************************")

# Calcul de l'erreur relative
err = 100 * (p_MC - p_ruine) / p_ruine
print("**************************************************")
print("L'erreur relative est de ", '{:.2f}'.format(err), "%")
print("pour ", NMC, " simulations Monte-Carlo")
print("**************************************************")

# Estimation ponctuelle de la durée moyenne du jeu
temps_moyen = temps_moyen/NMC

print("**************************************************")
print("La durée moyenne du jeu est estimée à :")
print("          ",'{:.3f}'.format(temps_moyen))
print("pour NMC =", NMC, "simulations Monte-Carlo.")
print("**************************************************")

#%%
# Quelques trajectoires pour le calcul de la probabilité de ruine b/(a+b)
a = 9     # fortune du joueur
b = 3     # gain cible 

# Méthode Monte-Carlo (estimation Monte-Carlo)
NMC = 5         # nombre de simulations Monte-Carlo

plt.figure(facecolor='w')

for k in range(NMC):
    rw = 0    
    path = [0]
    while (rw > -a) and (rw < b): 
            rw += rnd.choice([-1, 1])
            path.append(rw)
    plt.plot(path)

plt.axhline(y=b)
plt.axhline(y=-a)
plt.grid()
plt.xlabel('Temps')
plt.ylabel('Gain')
plt.title('Quelques trajectoires de jeux')
plt.show()   

###FIN######################################################################