import time
import requests
from flask import Flask, jsonify, request, Blueprint
import os
import numpy as np
import os
from FIWARE.set_status import set_status

print(os.getcwd())
print(os.listdir())
from PreferenceOptimization3 import GlispFinale
from ut import parse_values, get_values, gj
from FIWARE.set_status import set_status

print(os.getcwd())
print(os.listdir())

import json

app = Flask(__name__)



@app.route('/try')
def prova():
    return "It works"


@app.route('/task', methods=['POST', 'FETCH','GET'])
def get_x_next():
    if request.method=="GET":
        print("Get method received")
    print("Task received.")
    print(request.get_json())
    request_data = request.get_json()['data'][0]['status']

    set_status("running")
    acquisition_optimizer_type, b_best, b_in, fvarsX, fvarsY, x, delta = get_values(request_data)
    Y_best_A, x, x_A, y_b_in_A = parse_values(b_best, b_in, fvarsX, x)

    my_bo = GlispFinale.GLISpFinale(fvarsX, fvarsY, None,
                                    acquisition_optimizer_type=acquisition_optimizer_type,
                                    kfold=3, delta=delta, objective=(10, 1), batch_size=1,
                                    save_experiment=False, plot=False, title="", save=True,
                                    g=gj, name_opt='', plotAcquisition=False,
                                    X=x_A, Y_b_in=np.array([y_b_in_A]), Y_b_eq=np.array([[]]), Y_ind_best=Y_best_A
                                    )
    x_next = my_bo.run_optimization()[0]
    set_status("ready for input", x_next.tolist())

    # fiware
    resp = {
        "x_next": x_next.tolist()
    }
    # return jsonify(resp)
    print("resp" + str(resp))
    return resp


# @app.route('/get_x_next', methods=['POST'])
# def json_example():
#     request_data = request.get_json()
#     x_next = []
#     # fiware
#     resp = {
#         "x_next": x_next
#     }
#     return jsonify(resp)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
    set_status("ready for input", [])
