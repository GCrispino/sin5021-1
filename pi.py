import numpy as np
import datetime
from utils import bellman, evaluate


def PI(A, S, T, R, gamma, epsilon, epsilon_v):
    n_states = len(S)
    print 'n_states: ', n_states
    v = np.zeros(n_states, dtype="float64")
    # pi = np.full(n_states, "N")
    pi = np.zeros(n_states, dtype="uint8")

    k = 0

    # while (k < 10):
    while (True):
        print 'k: ', k
        newV = evaluate(T, R, v, A, pi, S, gamma, epsilon_v)
        # print 'hello: ', newV
        begin = datetime.datetime.now()
        # raw_input('vai calcular bellman...')
        newPi = np.fromiter((bellman(T, R, newV, A, S, s, gamma)[1]
                             for s in S), int)

        print datetime.datetime.now() - begin
        norm = np.linalg.norm(v - newV, np.inf)
        print("norm: ", norm, epsilon)
        if (norm < epsilon):
            break

        # if (np.all(pi == newPi)):
        #     break
        # print "pi: ", pi
        # print "newPi: ", newPi
        v = newV
        pi = newPi
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
        res = np.array([bellman(T, R, Vpi, A, S, s, gamma) for s in S])
        newPi = np.array([x[1] for x in res])
        if np.all(newPi == pi):
            break

        v = Vpi
        pi = np.copy(newPi)

        k += 1

    return pi, v, k
