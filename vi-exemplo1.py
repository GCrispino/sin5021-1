import numpy as np
import json
from vi import VI

matT = None
with open('matT.json', 'r') as f:
    matT = json.load(f)


def T(s, a, _s): return matT[s - 1][a.upper()][_s - 1]


def R(s, a, _s): return (0 if s == 5 else -1)


gamma = 1
epsilon = 10 ** -4

S = np.array(range(1, 11))

A = ["N", "S", "L", "O"]

pi, v, k = VI(A, S, T, R, gamma, epsilon)

print k
print v
print pi
