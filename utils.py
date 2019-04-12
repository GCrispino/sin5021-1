import numpy as np


def bellman(T, R, v, A, S, s, gamma):
    res = [float('-inf'), None]

    for a in A:
        q = sum([T(s, a, _s) * (R(s, a, _s) + gamma * v[_s - 1])
                 for _s in S])
        if (q > res[0]):
            res = [q, a]

    return res


def evaluate_1(T, R, v, pi, S, gamma, m):
    I = np.identity(len(S))


def evaluate(T, R, v, pi, S, gamma, epsilon):
    v_old = np.copy(v)
    v_new = np.zeros(len(S))
    exit_loop = False
    i = 0
    while True:
        for s in S:
            v_new[s - 1] = sum([
                T(s, pi[s - 1], _s) * (
                    R(s, pi[s - 1], _s) + gamma * v_old[_s - 1]
                ) for _s in S
            ])

        if np.linalg.norm(v_new - v_old, np.inf) < epsilon:
            break
        v_old = np.copy(v_new)
        i += 1
    return v_new
