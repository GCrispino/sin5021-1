import numpy as np

def bellman(T, R, v, A, S, s, gamma):
    res = [float('-inf'), None]

    for a in A:
        q = sum([T(s, a, _s) * (R(s, a, _s) + gamma * v[_s - 1])
                 for _s in S])
        if (q > res[0]):
            res = [q, a]

    return res




def evaluate(T, R, v, pi, S, gamma, m):
    v_old = np.copy(v)
    v_new = np.array(len(S))
    for s in S:
        v_new = sum([T(s, pi[s - 1], _s) * (R(s, pi[s - 1], _s) + gamma * v_old[_s - 1]) for _s in S])
        v_old[s - 1] = v_new[s - 1]
    return v_new

"""
const improve = (T, R, v, A, S, s, gamma) =>
  A.reduce(
    (acc, a) => {
      const [max] = acc;
      const q = S.reduce(
        //para cada s'...
        (sum, _s) => sum + T(s, a, _s) * (R(s, a, _s) + gamma * v[_s - 1]),
        0
      );

      // console.log(q > max ? [q, a] : acc);
      return q > max ? [q, a] : acc;
    },
    [-Infinity, undefined]
  );
"""

