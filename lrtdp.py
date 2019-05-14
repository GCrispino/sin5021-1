import numpy as np
from utils import bellman
from scipy.sparse import issparse

SOLVED = True

# POSSÍVEL GARGALO: 
#  Sempre que chama a função bellman ele é copiado

def sparse_or_array(s,a,T):
    Ts = T(s, a, None)
    # POSSÍVEL GARGALO
    Ts = Ts.toarray()[0] if issparse(Ts) else Ts

    return Ts

def get_next_state(S, T, s, a):
    # gets all of s probabilities when executing action a

    Ts = sparse_or_array(s,a,T)

    # pick new state with probability values given by Ts row
    return np.random.choice(S, p=Ts)


def get_states_from_s_a(s, a, S, T):
    # Save that to use when T(s, a, None) is csr_matrix
    # arr_s = T(s, a, None).toarray()[0]
    arr_s = sparse_or_array(s,a,T)
    # print("arr_s", np.argwhere(arr_s > 0))
    # POSSÍVEL GARGALO?
    states = np.argwhere(arr_s > 0)[0] + 1

    return states


def check_solved(s, v, pi, S, T, R, A, labels, res, epsilon, gamma):
    rv = True
    closed = []
    _open = [] if labels[s - 1] else [s]
    n_updates = 0

    while _open != []:
        s = _open.pop()
        closed.append(s)

        if res[s - 1] > epsilon:
            rv = False
            continue

        # get greedy action
        a = pi[s - 1]
        ss = get_states_from_s_a(s, a, S, T)

        for _s in ss:
            if (
                (not labels[_s - 1]) and
                (not _s in _open) and
                (not _s in closed)
            ):
                _open.append(_s)

    if rv:
        for s_ in closed:
            labels[s_ - 1] = SOLVED
    else:
        q, i_a = bellman(T, R, v, A, S, s, gamma)
        n_updates += len(A) 
        pi[s - 1] = A[i_a]
        res[s - 1] = np.abs(q - v[s - 1])
        v[s - 1] = q
    return rv, n_updates


def LRTDP(A, S, T, R, G, gamma, epsilon):
    n_states = len(S)
    n_actions = len(A)
    labels = np.zeros(n_states, dtype=bool)
    pi = np.full(n_states, A[0])
    v = np.zeros(n_states, dtype="float64")
    res = np.zeros(n_states, dtype="float64")  # residuals

    n_updates = 0

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

            # get greedy action
            a = pi[s - 1]

            # check if bellman function needs to be modified for LRTDP
            q, i_a = bellman(T, R, v, A, S, s, gamma)
            n_updates += n_actions
            pi[s - 1] = A[i_a]
            res[s - 1] = np.abs(q - v[s - 1])
            v[s - 1] = q

            s = get_next_state(S, T, s, a)

        while visited != []:
            s = visited.pop()
            if not check_solved(s, v, pi, S, T, R, A, labels, res, epsilon, gamma):
                break
        print("Solved labels: ",np.where(labels == 1)[0].size,n_states)
        print("Residual: ",res)
    return pi,v
