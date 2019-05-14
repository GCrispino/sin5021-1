import datetime
import numpy as np
from utils import bellman


def VI(A, S, T, R, gamma, epsilon):
    n_states = len(S)
    n_actions = len(A)
    v = np.zeros(n_states)
    pi = np.chararray(n_states)

    k = 0

    n_updates = 0
    while (True):
        print('k: ', k)

        begin = datetime.datetime.now()

        res = [bellman(T, R, v, A, S, s, gamma) for s in S]
        newV = np.fromiter((x[0] for x in res), float)
        pi = np.fromiter((A[x[1]] for x in res), 'U1')

        end = datetime.datetime.now()
        print('bellman time: ', end - begin)

        norm = np.linalg.norm(newV - v, np.inf)
        print('norm: ', norm)
        if (np.linalg.norm(newV - v, np.inf) < epsilon):
            break
        v = newV
        k += 1

    n_updates = k * (n_states * n_actions)
    return pi, v, k
