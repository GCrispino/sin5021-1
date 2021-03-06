import numpy as np
import json
import random
from pi import PI_mat

matT = None
with open('matT2.json', 'r') as f:
    matT = json.load(f)


def T(s, a, _s):
    return matT[s - 1][a.upper()][_s - 1]


def R(s, a, _s): return (0 if s == 6 else -1)


gamma = .6
epsilon = 10 ** -8
n_estados = 6

S = np.array(range(1, n_estados + 1))

A = ["N", "S", "L", "O"]

matR = np.array([
    {a: [R(s, a, _s) for _s in S] for a in A}
    for s in S
])

pi, v, k = PI_mat(A, S, T, matT, R, matR, gamma, epsilon)

print 'k: ', k
print 'v: ', v
print 'pi: ', pi
