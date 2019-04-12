import numpy as np
import json
import random
from utils import bellman, evaluate

matT = None
with open('matT2.json', 'r') as f:
    matT = json.load(f)


def T(s, a, _s):
    return matT[s - 1][a.upper()][_s - 1]


def R(s, a, _s): return (0 if s == 6 else -1)


gamma = .9
epsilon = 10 ** -4
epsilon_v = 10 ** -3

S = np.array(range(1, 7))

A = ["N", "S", "L", "O"]

v = np.zeros(6)
pi = np.full(6, "N")

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

print 'k: ', k
print 'v: ', v
print 'pi: ', pi
