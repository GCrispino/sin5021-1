import numpy as np
from utils import bellman


def VI(A, S, T, R, gamma, epsilon):
    n_estados = len(S)
    v = np.zeros(n_estados)
    pi = np.chararray(n_estados)

    raw_input('criou os arrays...')
    k = 0

    while (True):
        print 'k: ', k
        res = [bellman(T, R, v, A, S, s, gamma) for s in S]

        newV = np.fromiter((x[0] for x in res), float)
        pi = np.fromiter((A[x[1]] for x in res), 'S1')
        norm = np.linalg.norm(newV - v, np.inf)
        print 'norm: ', norm
        if (np.linalg.norm(newV - v, np.inf) < epsilon):
            break
        v = newV
        k += 1

    return pi, v, k
