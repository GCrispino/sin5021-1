import numpy as np
from utils import bellman

def get_next_state(S,T,s,a):
    # gets all of s probabilities when executing action a
    Ts = T(s,a,None)

    # pick new state with probability values given by Ts row
    return np.random.choice(S,p=Ts)


def LRTDP(A, S, T, R, G, gamma, epsilon):
    SOLVED = True
    n_states = len(S)
    labels = np.zeros(n_states,dtype=bool)
    pi = np.full(n_states, A[0])
    v = np.zeros(n_states, dtype="float64")
    res = np.zeros(n_states, dtype="float64")

    # while initial state is not labeled as solved
    while labels[0] != SOLVED:
        s = S[0]
        """ 
            maybe this can be faster by using
            a numpy array of size n_states (might consume too much memory though)
        """
        visited = []

        # while state s is not labeled as solved
        # HAVE TO CHECK HERE IF STATES ARE 0 OR 1-INDEXED
        while labels[s - 1] != SOLVED:
            visited.append(s)
            if s in G:
                break

            a = pi[s - 1]

            # check if bellman function needs to be modified for LRTDP
            q,_ = bellman(T,R,v,A,S,s,gamma)
            v[s - 1] = q

            s = get_next_state(S,T,s,a)

        while visited != []:
            # MISSING IMPLEMENTATION
            pass
    return pi 

