import numpy as np
import json
from utils import bellman

matT = None
with open('matT.json', 'r') as f:
    matT = json.load(f)

def T(s, a, _s): return matT[s - 1][a.upper()][_s - 1]


def R(s, a, _s): return (0 if s == 5 else -1)


gamma = 1
epsilon = 10 ** -10

S = np.array(range(1, 11))

A = ["N", "S", "L", "O"]

v = np.zeros(10)
pi = np.chararray(10)

k = 0

while (True):
    res = [bellman(T, R, v, A, S, s, gamma) for s in S]

    newV = np.array([x[0] for x in res])
    pi =  np.array([x[1] for x in res])
    if (np.linalg.norm(newV - v,np.inf) < epsilon):
       break
    v = newV
    k += 1

print v
print pi
