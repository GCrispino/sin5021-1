import numpy as np
import pandas as pd
import sys
import datetime


def read_rewards(path):
    return pd.read_csv(path, header=None).values.T[0]


"""
    for each pair (action: file_path), execute read_action function
    and reduce all to (n_states,n_actions,n_states) array
"""


def get_n_states(path):
    df = pd.read_csv(path, sep='   ', header=None, engine='python')
    return int(df[0].max())


def read_actions1(paths_dict):
    actions = paths_dict.keys()
    action_mats = {a: read_action(path) for a, path in paths_dict.items()}
    n_states = len(action_mats[actions[0]][0])
    res_mat = [{} for i in range(n_states)]

    for i_s in range(0, n_states):
        for a in actions:
            res_mat[i_s][a] = action_mats[a][i_s]

    return res_mat


def read_actions(paths):
    n_actions = len(paths)

    n_states = get_n_states(paths[0])

    action_mats = np.array([read_action(paths[i])
                            for i in range(0, n_actions)])

    return action_mats.transpose(1, 0, 2)


"""
    given path, read file on this location and
    construct (n_actions,n_states,n_states)
"""


def read_action(path):
    print("Lendo dados de acao do arquivo ", path)

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
    res = (float('-inf'), 0)

    # print 'oi: ', T(0, A[0], None), R(0, None, None), v
    # print T(0, A[0], None).shape, R(0, None, None).shape, v.shape
    # sys.exit(0)

    for i in np.arange(0, len(A)):
        q = np.sum(
            T(s, A[i], None).dot(R(s, None, None) + gamma * v)
        )
        if (q > res[0]):
            res = (q, i)

    return res


def evaluate_1(T, R, v, pi, S, gamma, m):
    I = np.identity(len(S))


def evaluate(T, R, v, A, pi, S, gamma, epsilon):
    v_old = np.copy(v)
    v_new = np.zeros(len(S))
    i = 0
    while True:
        # print 's begin: '
        begin = datetime.datetime.now()

        for s in S:
            try:
                w = s - 1
                x = v_new[s - 1]
                y = pi[s - 1]
                z = A[pi[s - 1]]
            except:
                print 'xiiii: '
                print A, A[-1], type(A)
                print pi
                print s - 1, v_new[s - 1], pi[s - 1], A[pi[s - 1]]
                sys.exit(0)
            v_new[s - 1] = T(s, A[pi[s - 1]], None).dot(
                R(s, None, None) + gamma * v_old
            )
        # end = datetime.datetime.now()
        # print 's end: '
        # print 'time: ', end - begin
        norm = np.linalg.norm(v_new - v_old, np.inf)
        # print("  norm_eval: ", norm, epsilon)
        # if (np.sum((np.fabs(v_old - v_new))) <= epsilon):
        if norm < epsilon:
            break
        v_old = np.copy(v_new)
        i += 1
    return v_new
