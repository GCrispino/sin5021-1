
import json
import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import utils
from pi import PI
from vi import VI

GRID_WIDTH = 9
GRID_HEIGHT = 15

if len(sys.argv) < 3:
    sys.exit("USAGE: python main.py <env_folder> <algorithm={0,1,2}>")

env_path = sys.argv[1]
algorithm = int(sys.argv[2])

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


paths = [
    env_path + '/Action01.txt',
    env_path + '/Action02.txt',
    env_path + '/Action03.txt',
    env_path + '/Action04.txt',
    env_path + '/Action05.txt',
    env_path + '/Action06.txt',
]


print("Lendo arquivos de acoes e recompensa...")
begin = datetime.datetime.now()
a_mat = utils.read_actions(paths)
end = datetime.datetime.now()
# if (a_mat.shape):
#     print a_mat[15][0]
print("Time spent: ", str(end - begin))


def R(s, a, _s):
    if (not s) and (not a) and (not _s):
        return rewards
    return rewards[s - 1]


A = ["N", "S", "L", "O", "U", "D"]


def T(s, a, _s):
    i_a = A.index(a.upper())
    return a_mat[s - 1][i_a] if _s == None else a_mat[s - 1][i_a][_s - 1]


gamma = .9
epsilon = 10 ** -10
epsilon_v = 10 ** -7

S = np.arange(1, len(a_mat) + 1)

if (algorithm == 0):
    pi, v, k = VI(A, S, T, R, gamma, epsilon)
elif (algorithm == 1):
    pi, v, k = PI(A, S, T, R, gamma, epsilon, epsilon_v)
# print pi.shape, v.shape

pi = pi.reshape((len(S) / (GRID_WIDTH * GRID_HEIGHT), GRID_WIDTH, GRID_HEIGHT))
v = v.reshape((len(S) / (GRID_WIDTH * GRID_HEIGHT), GRID_WIDTH, GRID_HEIGHT))

for floor in v:
    plt.figure()
    plt.imshow(floor)

# print pi, v
# print pi.shape, v.shape

with open('result.json', 'w') as fp:
    json.dump([pi.tolist(), v.tolist()], fp, indent=2)


plt.show()
