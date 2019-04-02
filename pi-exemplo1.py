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
epsilon = 10 ** -10

S = np.array(range(1, 11))

A = ["N", "S", "L", "O"]

v = np.zeros(10)
pi = np.full(10, "N")

k = 0

while (True):
    print 'k: ', k
    newV = np.array(evaluate(T, R, v, pi, S, gamma, 20))
    res = [bellman(T, R, v, A, S, s, gamma) for s in S]

    newPi = np.array([x[1] for x in res])
    # if (k > 0 and np.array_equal(pi, newPi)):
    #     print 'IGUAL!!!!!!'
    if (np.linalg.norm(newV - v, np.inf) < epsilon):
        # if (k > 3):
        break
    v = newV
    pi = newPi
    k += 1
    print 'v: ', v
    print 'pi: ', pi

print k
print v
print pi
