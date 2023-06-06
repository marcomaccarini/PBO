import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def parse_values(b_best, b_in, fvarsX, x):
    x_A = np.empty((0, len(fvarsX)))
    if x != "":
        x_A = np.vstack((x_A, x))
    y_b_in_A = [np.empty((0, 1)) for i in range(len(b_in))]
    for i in range(len(y_b_in_A)):
        y_b_in_A[i] = np.array(b_in[i])
    Y_best_A = [np.empty((0, 1)) for i in range(len(b_best))]
    if b_best != "":
        for i in range(len(Y_best_A)):
            # Y_best_init[i] = np.array(json.loads(optimization.Y_best_init)[i])
            Y_best_A[i] = int(b_best[i])
    for x in fvarsX:
        if 'step format' in x['type']:
            x['type'] = 'discrete'
            domain = [float(k) for k in x['domain'].split(",")]
            step = domain[2]
            start = domain[0]
            stop = domain[1]
            domain = list(map(float, np.arange(start, stop, step)))
            x['domain'] = domain
        elif 'custom' in x['type']:
            x['type'] = 'discrete'
            domain = [float(k) for k in x['domain'].split(",")]
            x['domain'] = domain

    return Y_best_A, x, x_A, y_b_in_A


def get_values(request_data):
    fvarsX = request_data['fvarsX']
    fvarsY = request_data['fvarsY']
    acquisition_optimizer_type = request_data['acquisition_optimizer_type']
    x = request_data['x']
    b_in = request_data['b_in']
    b_best = request_data['b_best']
    # exploration = bool(request_data['exploration'])
    delta = float(request_data['delta'])
    return acquisition_optimizer_type, b_best, b_in, fvarsX, fvarsY, x, delta


def gj(xt, Y, Models):
    (N, n) = np.shape(Y)
    if n == 1:
        return Y[:, 0]
    if n == 2:
        return Y[:, 0] + Y[:, 1]
    if n == 3:
        return Y[:, 0] + Y[:, 1] + Y[:, 2]
    if n == 4:
        return Y[:, 0] + Y[:, 1] + Y[:, 2] + Y[:, 3]
    if n == 5:
        return Y[:, 0] + Y[:, 1] + Y[:, 2] + Y[:, 3] + Y[:, 4]
    if n == 6:
        return Y[:, 0] + Y[:, 1] + Y[:, 2] + Y[:, 3] + Y[:, 4] + Y[:, 5]


j = {

    "status": {
        "value": {
            "fvarsX": [
                {
                    "name": "a",
                    "type": "continuous",
                    "domain": [
                        -200.0,
                        200.0
                    ]
                },
                {
                    "name": "b",
                    "type": "continuous",
                    "domain": [
                        -200.0,
                        200.0
                    ]
                }
            ],
            "fvarsY": [
                {
                    "name": "out1",
                    "type": "continuous",
                    "domain": []
                }
            ],
            "acquisition_optimizer_type": "lbfgsb",
            "x": [
                [
                    34, 32
                ],
                [
                    12, 65
                ],
                [
                    8, 43
                ],
                [
                    12, 12
                ],
                [
                    42, 24
                ],
                [
                    18, 30
                ]

            ],
            "b_in": [
                [
                    0,
                    1,
                    1
                ],
                [
                    2,
                    1,
                    1
                ],
                [
                    3,
                    2,
                    1
                ],
                [
                    3,
                    4,
                    1
                ],
                [
                    3,
                    5,
                    1
                ]
            ],
            "b_best": [
                3
            ],
            "delta": 3
        }
    }
}


def get_msg():
    return j
