import numpy as np
import json
import random
import sys
from pi import PI

matT = None
with open('matT.json', 'r') as f:
    matT = json.load(f)


def T(s, a, _s):
    return np.array(matT[s - 1][a.upper()]) if _s == None else matT[s - 1][a.upper()][_s - 1]


arrR = np.full(10, -1)
arrR[4] = 0


def R(s=None, a=None, _s=None):
    if (not s) and (not a) and (not _s):
        return arrR
    return (0 if s == 5 else -1)


gamma = .999
epsilon = 10 ** -5
epsilon_v = 10 ** -4

S = np.array(range(1, 11))

A = ["N", "S", "L", "O"]

pi, v, k, _ = PI(A, S, T, R, gamma, epsilon, epsilon_v)

print('k: ', k)
print('v: ', v)
print('pi: ', pi)
