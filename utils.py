import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import sys
import datetime


def read_rewards(path):
    return pd.read_csv(path, header=None).values.T[0]


def get_n_states(path):
    df = pd.read_csv(path, sep='   ', header=None, engine='python')
    return int(df[0].max())


def read_actions2(paths_dict):
    actions = paths_dict.keys()
    action_mats = {a: read_action(path) for a, path in paths_dict.items()}
    n_states = len(action_mats[actions[0]][0])
    res_mat = [{} for i in range(n_states)]

    for i_s in range(0, n_states):
        for a in actions:
            res_mat[i_s][a] = action_mats[a][i_s]

    return res_mat


def read_actions(paths):
    """
        for each pair (action: file_path), execute read_action function
        and reduce all to (n_states,n_actions,n_states) array
    """
    n_actions = len(paths)

    action_mats = np.array([read_action(paths[i])
        for i in range(0, n_actions)], dtype="float16")
    n_states = len(action_mats[0][0])

    return [
            csr_matrix([
                action_mats[a][i_s] for a in range(0, n_actions)
                ], dtype="float16") for i_s in range(0, n_states)
            ]


    # Tentar retornar um dataframe e depois agregar na read_functions
def read_action(path):
    """
        given path, read file on this location and
        construct (n_actions,n_states,n_states)
    """
    print("Reading action data from file ", path)

    df = pd.read_csv(path, sep='   ', header=None, engine='python')
    max_state = int(df[0].max())
    min_state = int(df[0].min())
    a_mat = np.zeros((max_state, max_state), dtype='float16')

    # for s in np.arange(min_state, max_state + 1):
    for s in np.arange(min_state, max_state + 1):
        df_s = df[df[0] == s]
        for _, row in df_s.iterrows():
            _s = int(row[1])
            p = row[2]
            a_mat[s - 1][_s - 1] = p
    print("Finished ", path, "!")

    return a_mat


def bellman(T, R, v, A, S, s, gamma):
    res = (float('-inf'), 0)

    # begin = datetime.datetime.now()

    for i in np.arange(0, len(A)):
        q = T(s, A[i], None).dot(R(s, None, None) + gamma * v)
        if (q > res[0]):
            res = (q, i)

    # SOLUÇÃO ALTERNATIVA:
    # depois tentar fromfunction
    # res = np.fromiter((np.sum(
    #     T(s, A[i], None).dot(R(s, None, None) + gamma * v)
    # ) for i in np.arange(0, len(A))), float)
    # max_a = np.argmax(res)

    # return (res[max_a], max_a)
    # end = datetime.datetime.now()
    # print('bellman inside time for state ', s, ': ', end - begin)

    return res



def evaluate(T, R, v, A, pi, S, gamma, epsilon):
    n_states = len(S)
    v_old = np.copy(v)
    v_new = np.zeros(n_states)
    i = 0

    while True:
        begin = datetime.datetime.now()

        for s in S:
            v_new[s - 1] = T(s, pi[s - 1], None).dot(
                R(s, None, None) + gamma * v_old
            )

        norm = np.linalg.norm(v_new - v_old, np.inf)
        if norm < epsilon:
            break
        v_old = v_new

        i += 1

    return v_new, i


base_arrow = {
    'width': 0.04,
    'head_width': 0.3,
    'head_length': 0.2,
    'color': "white"
}

arrows = {
    'N': {
        'x': 0,
        'dx': 0,
        'y': 0.4,
        'dy': -0.6,
        **base_arrow
    },
    'S': {
        **base_arrow,
        'x': 0,
        'dx': 0,
        'y': -0.4,
        'dy': 0.6
    },
    'L': {
        **base_arrow,
        'x': -0.4,
        'dx': 0.6,
        'y': 0,
        'dy': 0
    },
    'O': {
        **base_arrow,
        'x': 0.4,
        'dx': -0.6,
        'y': 0,
        'dy': 0
    },
    'U': {
        **base_arrow,
        'x': 0,
        'dx': 0,
        'y': 0.1,
        'dy': -0.001,
        'head_width': 0.35,
        'head_length': 0.25,
    },
    'D': {
        **base_arrow,
        'x': 0,
        'dx': 0,
        'y': -0.1,
        'dy': 0,
        'head_width': 0.35,
        'head_length': 0.25,
    }
}


def plotArrow(action, i, j, A, plt):
    arrow = arrows[action]
    plt.arrow(
        arrow['x'] + j,
        arrow['y'] + i,
        arrow['dx'], arrow['dy'],
        width=arrow['width'],
        head_width=arrow['head_width'], head_length=arrow['head_length'], color=arrow['color']
    )
