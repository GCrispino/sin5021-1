import numpy as np
from utils import bellman, evaluate


def PI(A, S, T, R, gamma, epsilon, epsilon_v):
    n_states = len(S)
    v = np.zeros(n_states)
    pi = np.full(n_states, "N")

    k = 0

    while (True):
        newV = np.copy(evaluate(T, R, v, pi, S, gamma, epsilon_v))
        res = [bellman(T, R, v, A, S, s, gamma) for s in S]

        newPi = np.array([x[1] for x in res])
        if (np.linalg.norm(newV - v, np.inf) < epsilon):
            break
        v = newV
        pi = np.copy(newPi)
        k += 1

    return pi, v, k


def PI_mat(A, S, T, matT, R, matR, gamma, epsilon):
    n_estados = len(S)
    v = np.zeros(n_estados)
    pi = np.full(n_estados, "N")

    k = 0

    while True:
        Tpi = np.array([
            matT[s - 1][pi[s - 1]]
            for s in S
        ])

        Rpi = np.array([
            sum([matR[s - 1][pi[s - 1]][_s - 1] * T(s, pi[s - 1], _s)
                 for _s in S])
            for s in S
        ]).T
        # policy evaluation
        Vpi = np.dot(np.linalg.inv(np.eye(len(S)) - gamma * Tpi), Rpi)
        # policy improvement
        res = [bellman(T, R, Vpi, A, S, s, gamma) for s in S]
        newPi = np.array([x[1] for x in res])
        if np.all(newPi == pi):
            break

        v = Vpi
        pi = np.copy(newPi)

        k += 1

    return pi, v, k
