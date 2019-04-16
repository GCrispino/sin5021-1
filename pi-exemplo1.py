import numpy as np
import json
import random
from pi import PI

matT = None
with open('matT.json', 'r') as f:
    matT = json.load(f)


def T(s, a, _s = None):
    return matT[s - 1][a.upper()][_s - 1] if _s == None else matT[s - 1][a.upper()][_s - 1]

def T(s, a):
    return matT[s - 1][a.upper()]



def R(s, a, _s): return (0 if s == 5 else -1)


gamma = .9
epsilon = 10 ** -4
epsilon_v = 10 ** -3

S = np.array(range(1, 11))

A = ["N", "S", "L", "O"]

pi, v, k = PI(A, S, T, R, gamma, epsilon, epsilon_v)

print 'k: ', k
print 'v: ', v
print 'pi: ', pi
