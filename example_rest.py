import requests
from ut import send_j
import json

url = "http://127.0.0.1:5000/get_x_next"


def f(x):
    return x[0] ** 2 +x[1]*2


def pref(x_next, j):
    bb = j['x'][j['b_best'][0]]
    id_ = len(j['x'])
    j['x'].append(x_next)
    v_bb = f(bb)
    v_x_next = f(x_next)
    if v_bb < v_x_next:
        j['b_in'].append([j['b_best'][0], id_, 1])
    else:
        j['b_in'].append([id_, j['b_best'][0], 1])
        j['b_best'][0] = id_

    # a = input('-1 if x next is better than the best, 1 otherwise')
    # if '-1':
    #    j['b_in'].append([id_, j['b_best'][0] , 1])
    #    j['b_best'][0] = id_
    # else:
    #    j['b_in'].append([j['b_best'][0] , id_, 1])


j = send_j()
x_next = requests.post(url, json=j)
x_next = x_next.json().get('x_next')
print(x_next)
iterations = 5

for i in range(iterations):
    print(j['b_in'])
    pref(x_next, j)
    print(j['b_in'])
    x_next = requests.post(url, json=j)
    print(x_next.json())
    x_next = x_next.json().get('x_next')
    print(x_next)


