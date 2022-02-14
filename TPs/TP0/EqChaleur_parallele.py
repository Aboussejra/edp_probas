#%%
# Vérification de la formule de représentation probabiliste de la solution

import multiprocessing as mp
import sys
import numpy as np

from datetime import datetime


def thetaExact1(x, y):
    return 20 * x * y


# Générateur de marche aléatoire sur un réseau (Random Walk)
def RW(position, pas):
    direction = np.random.randint(low=1, high=5, size=1)
    if direction == 1:
        step = pas * np.array([1.0, 0])
    elif direction == 2:
        step = pas * np.array([0, 1.0])
    elif direction == 3:
        step = pas * np.array([-1.0, 0])
    else:
        step = pas * np.array([0, -1.0])
    return position + step


def thetaMC(NMC):
    thetaMC = []
    # Re-seed the random number generator
    np.random.seed()

    for nMC in range(NMC):
        pointCourant = np.array([0.5, 0.5])
        pointSuivant = RW(pointCourant, h)
        pointCourant, pointSuivant = pointSuivant, RW(pointSuivant, h)
        while (0 < pointSuivant[0] < 1) & (0 < pointSuivant[1] < 1):
            pointCourant, pointSuivant = pointSuivant, RW(pointSuivant, h)
        thetaMC.append(thetaExact1(pointSuivant[0], pointSuivant[1]))

    thetaHat = np.mean(thetaMC)
    return thetaHat


def success_handler(result):
    print("Call was a Sucess, found theta hat on this CPU", result)
    theta_hat_per_processes.append(result)


def error_handler(e):
    print("Error", file=sys.stderr)
    print(e.__cause__, file=sys.stderr)


if __name__ == "__main__":
    N = 32
    h = 1 / N
    NMC = 2000
    print(f"Number of simulations is {NMC}")
    theta_hat_per_processes = []
    CPUs = mp.cpu_count()
    print(f"I have {CPUs}  CPUs")
    pool = mp.Pool(CPUs)
    start_time_parallel = datetime.now()
    for i in range(CPUs):
        pool.apply_async(
            thetaMC,
            args=(int(NMC / CPUs),),
            callback=success_handler,
            error_callback=error_handler,
        )
    pool.close()
    pool.join()
    print(f"Theta across processes is {np.mean(theta_hat_per_processes)} ")
    time_elapsed_parallel = datetime.now() - start_time_parallel
    print("Time elapsed parallel (hh:mm:ss.ms) {}".format(time_elapsed_parallel))

    start_time_sequential = datetime.now()
    res = thetaMC(NMC)
    time_elapsed_sequential = datetime.now() - start_time_sequential
    print("Time elapsed sequential (hh:mm:ss.ms) {}".format(time_elapsed_sequential))
