import numpy as np
import json
import random
from utils import bellman, evaluate

matT = None
with open('matT.json', 'r') as f:
    matT = json.load(f)


def T(s, a, _s):
    return matT[s - 1][a.upper()][_s - 1]


def R(s, a, _s): return (0 if s == 5 else -1)


gamma = .9
epsilon = 10 ** -4
epsilon_v = 10 ** -3

S = np.array(range(1, 11))

A = ["N", "S", "L", "O"]

matR = np.array([
    {a: [R(s, a, _s) for _s in S] for a in A}
    for s in S
])

v = np.zeros(10)
pi = np.full(10, "N")

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
    # print newPi, pi
    if np.all(newPi == pi):
        break

    v = Vpi
    pi = np.copy(newPi)

    k += 1

print 'k: ', k
print 'v: ', v
print 'pi: ', pi
