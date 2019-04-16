import numpy as np
import pandas as pd


def read_rewards(path):
    return pd.read_csv(path, header=None).values


""" 
    for each pair (action: file_path), execute read_action function
    and reduce all to (n_states,n_actions,n_states) array
"""


def read_actions(paths_dict):
    actions = paths_dict.keys()
    action_mats = {a: read_action(path) for a, path in paths_dict.items()}
    n_states = len(action_mats[actions[0]][0])
    res_mat = [{} for i in range(n_states)]

    for i_s in range(0, n_states):
        for a in actions:
            res_mat[i_s][a] = action_mats[a][i_s].tolist()

    return res_mat


"""
    given path, read file on this location and
    construct (n_actions,n_states,n_states)
"""


def read_action(path):
    df = pd.read_csv(path, sep='   ', header=None, engine='python')
    max_state = int(df[0].max())
    min_state = int(df[0].min())
    a_mat = np.zeros((max_state, max_state))

    for s in range(min_state, max_state + 1):
        df_s = df[df[0] == s]
        for _, row in df_s.iterrows():
            _s = int(row[1])
            p = row[2]
            a_mat[s - 1][_s - 1] = p

    return a_mat


def bellman(T, R, v, A, S, s, gamma):
    res = [float('-inf'), None]

    for a in A:
        q = sum(
            [
                0 if T(s, a, _s) == 0
                else T(s, a, _s) * (R(s, a, _s) + gamma * v[_s - 1])
                for _s in S
            ]
        )
        if (q > res[0]):
            res = [q, a]

    return res


def evaluate_1(T, R, v, pi, S, gamma, m):
    I = np.identity(len(S))


def evaluate(T, R, v, pi, S, gamma, epsilon):
    v_old = np.copy(v)
    v_new = np.zeros(len(S))
    i = 0
    while True:
        for s in S:
            v_new[s - 1] = sum([
                0 if T(s, pi[s - 1], _s) == 0
                else T(s, pi[s - 1], _s) * (
                    R(s, pi[s - 1], _s) + gamma * v_old[_s - 1]
                )
                for _s in S
            ])
        norm = np.linalg.norm(v_new - v_old, np.inf)
        if norm < epsilon:
            break
        v_old = np.copy(v_new)
        i += 1
    return v_new
