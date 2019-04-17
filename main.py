import json
import datetime
import sys
import numpy as np
import utils
from pi import PI
from vi import VI

if len(sys.argv) == 1:
    sys.exit("USAGE: python main.py <env_folder>")

env_path = sys.argv[1]

rewards_path = env_path + '/Rewards.txt'
rewards = utils.read_rewards(rewards_path)

paths_dict = {
    'N': env_path + '/Action01.txt',
    'S': env_path + '/Action02.txt',
    'L': env_path + '/Action03.txt',
    'O': env_path + '/Action04.txt',
    'U': env_path + '/Action05.txt',
    'D': env_path + '/Action06.txt',
}


begin = datetime.datetime.now()
a_mat = utils.read_actions(paths_dict)
end = datetime.datetime.now()
print("Time spent: ", str(end - begin))


def R(s, a, _s):
    if (not s) and (not a) and (not _s):
        return rewards
    return rewards[s - 1]


def T(s, a, _s):
    return np.array(a_mat[s - 1][a.upper()]) if _s == None else a_mat[s - 1][a.upper()][_s - 1]


gamma = 0.9
epsilon = 10 ** -4
epsilon_v = 10 ** -3

S = np.arange(1, len(a_mat) + 1)

A = ["N", "S", "L", "O", "U", "D"]


pi, v, k = PI(A, S, T, R, gamma, epsilon, epsilon_v)
#pi, v, k = VI(A, S, T, R, gamma, epsilon)

print pi, v


with open('result.json', 'w') as fp:
    json.dump([pi.tolist(), v.tolist()], fp)
