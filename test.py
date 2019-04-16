import json
import utils
import datetime

paths_dict = {
    'N': 'Dados/Ambiente1/Action01.txt',
    'S': 'Dados/Ambiente1/Action02.txt',
    'L': 'Dados/Ambiente1/Action03.txt',
    'O': 'Dados/Ambiente1/Action04.txt',
    'U': 'Dados/Ambiente1/Action05.txt',
    'D': 'Dados/Ambiente1/Action06.txt',
}

begin = datetime.datetime.now()
a_mat = utils.read_actions(paths_dict)
end = datetime.datetime.now()
print("Time spent: ", str(end - begin))

with open('result.json', 'w') as fp:
    json.dump(a_mat, fp)
