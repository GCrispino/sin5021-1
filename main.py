
import json
import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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


print("Reading rewards and actions files...")
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


timestamp = datetime.datetime.now().timestamp()

gamma = .99
epsilon = 10 ** -10
epsilon_v = 10 ** -7

S = np.arange(1, len(a_mat) + 1)

begin = datetime.datetime.now()
total_inner_iterations = None
if (algorithm == 0):
    pi, v, k = VI(A, S, T, R, gamma, epsilon)
elif (algorithm == 1):
    pi, v, k, total_inner_iterations = PI(
        A, S, T, R, gamma, epsilon, epsilon_v
    )
# print pi.shape, v.shape
end = datetime.datetime.now()
time = end - begin

print("Time spent: ", str(time))

pi = pi.reshape(
    (int(len(S) / (GRID_WIDTH * GRID_HEIGHT)), GRID_WIDTH, GRID_HEIGHT))
v = v.reshape((int(len(S) / (GRID_WIDTH * GRID_HEIGHT)),
               GRID_WIDTH, GRID_HEIGHT))

pp = PdfPages('./results/result' + str(timestamp) + '.pdf')

for i_f in range(len(v)):
    floor_v = v[i_f]
    floor_pi = pi[i_f]
    plt.figure()

    n_rows, n_columns = floor_v.shape
    for i in range(n_rows):
        for j in range(n_columns):
            action = floor_pi[i][j]
            utils.plotArrow(action, i, j, A, plt)

    plt.imshow(floor_v)
    plt.savefig(pp, format="pdf")

pp.close()

# print pi, v
# print pi.shape, v.shape

with open('./results/result' + str(timestamp) + '.json', 'w') as fp:
    res_dict = {
        'alg': algorithm,
        'k': k,
        'gamma': gamma,
        'epsilon': epsilon,
        'epsilon': epsilon,
        'epsilon_v': epsilon_v,
        'time': time.total_seconds(),
        'pi': pi.tolist(),
        'v': v.tolist(),
    }
    if (total_inner_iterations):
        res_dict['inner_iterations'] = total_inner_iterations
    json.dump(res_dict, fp, indent=2)

# plt.show()
