import numpy as np
from utils import bellman


def VI(A, S, T, R, gamma, epsilon):
    n_estados = len(S)
    v = np.zeros(n_estados)
    pi = np.chararray(n_estados)

    k = 0

    while (True):
        res = [bellman(T, R, v, A, S, s, gamma) for s in S]

        newV = np.array([x[0] for x in res])
        pi = np.array([x[1] for x in res])
        if (np.linalg.norm(newV - v, np.inf) < epsilon):
            break
        v = newV
        k += 1

    return pi, v, k
