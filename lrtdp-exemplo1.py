import numpy as np
import json
import random
import sys
from lrtdp import LRTDP

matT = None
with open('matT.json', 'r') as f:
    matT = json.load(f)

A = ["N", "S", "L", "O"]


def T(s, a, _s):
    return np.array(matT[s - 1][a.upper()]) if _s == None else matT[s - 1][a.upper()][_s - 1]


arrR = np.full(10, -1)
arrR[4] = 0


def R(s=None, a=None, _s=None):
    if (not s) and (not a) and (not _s):
        return arrR
    return (0 if s == 5 else -1)


gamma = .9
epsilon = 10 ** -5

S = np.array(range(1, 11))

G = [5]

pi,v = LRTDP(A, S, T, R, G, gamma, epsilon)

# print('k: ', k)
print('v: ', v)
print('pi: ', pi)
