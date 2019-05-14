import numpy as np
import datetime
from utils import bellman, evaluate


def PI(A, S, T, R, gamma, epsilon, epsilon_v):
    n_states = len(S)
    n_actions = len(A)
    v = np.zeros(n_states, dtype="float64")
    pi = np.full(n_states, A[0])

    k = 0
    n_updates = 0

    total_inner_iterations = 0
    while (True):
        print('k: ', k)
        newV, inner_iterations = evaluate(T, R, v, A, pi, S, gamma, epsilon_v)
        n_evaluate_updates = n_states * inner_iterations
        total_inner_iterations += inner_iterations
        begin = datetime.datetime.now()
        newPi = np.fromiter((A[bellman(T, R, newV, A, S, s, gamma)[1]]
                             for s in S), 'U1')
        n_improvement_updates = n_states * n_actions
        n_updates += n_evaluate_updates + n_improvement_updates
        print(datetime.datetime.now() - begin)
        norm = np.linalg.norm(v - newV, np.inf)
        print("norm: ", norm, epsilon)
        if (norm < epsilon):
            break

        # if (np.all(pi == newPi)):
        #     print('EQUAL!')
            # break

        v = newV
        pi = newPi
        k += 1
    #n_updates = n_states * total_inner_iterations

    return pi, v, k, total_inner_iterations


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
        res = np.array([bellman(T, R, Vpi, A, S, s, gamma) for s in S])
        newPi = np.array([x[1] for x in res])
        if np.all(newPi == pi):
            break

        v = Vpi
        pi = np.copy(newPi)

        k += 1

    return pi, v, k
