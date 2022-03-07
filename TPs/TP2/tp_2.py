#%%

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

num = 1000
times = np.linspace(0, A1,num=num)
NMC = 100

def fourier_wiener(times)
    b_t = np.random.rand(1)*times
    # 2000 harmoniques
    for i in range(1,1000):
        b_t += np.random.randn(1)*(np.sqrt(2)/(i*np.pi))*np.sin(np.pi*i*times)
    return b_t

# fourier_wiener_times = []
# for i in range(NMC):
#     fourier_wiener_times.append(fourier_wiener(times))
    
# #density_estimation = stats.gaussian_kde(fourier_wiener_times)
# MCsim = np.mean(fourier_wiener_times,axis=0)
sim = fourier_wiener(times)
plt.plot(times, fourier_wiener(times))
plot_acf(np.diff(sim),lags=50)

# Test densité normale, ACF
# On a un mouvement brownien standard sur [0;T] 
# COmparaison: 
    
def Rwalk(N, p):
   "Simulation d'une marche aléatoire sur une grille"   
   pas = 2*np.random.binomial(n=1, p=p, size=N) - 1
   path = np.sqrt(1/N)*np.cumsum(pas)
   return path

NbSimu = 1 # nombre de trajectoires simulées
mb = Rwalk(num, 0.5)
plot_acf(np.diff(mb))
temps = np.linspace(start=0, stop=1, num=num)
plt.figure(facecolor='w')
plt.plot(temps,mb)
plt.plot(temps,sim,color='red')
plt.plot(temps,2*np.sqrt(temps),'k--')
plt.plot(temps,-2*np.sqrt(temps),'k--')
plt.title("Comparaison MB et approximation par processus de Wiener")
plt.xlabel('Temps')
plt.ylabel('Position')
plt.grid()
plt.show()


# S_zero = 100
# T = 1
# mu = 0.1
# sigma = 0.5
# # Définissons un pas de temps
# N = 100
# h = T/N
# S_analytique_T_list = []
# # Générateur de Brownien standard
# def Brownien(t):
#     return np.sqrt(t) * np.random.normal(size=1)

# for i in range(NMC):
#     S_analytique_T_list.append(S_zero*np.exp(((mu-sigma**2)/2)*T + sigma*Brownien(T))[0])
# true_density = stats.gaussian_kde(S_analytique_T_list) 
# S_analytique_T_analy_MC = np.mean(S_analytique_T_list)

# def simul_euler(T):
#     S_T = S_zero
#     t = 0
#     for i in range(N):
#         S_T = S_T + mu*S_T*h + sigma*S_T*(Brownien(t+h)-Brownien(t))
#         t = t+h
#     return S_T
# print(simul_euler(T))
