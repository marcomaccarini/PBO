import GlispFinale as GL
import numpy as np
from utils.preference_functions import compute_preference
# from rosFolder import talkerPreferencesMultiArray as t
import matplotlib.pyplot as plt
import math
import utils.process
import utils.math
from datetime import datetime


def point_eval_out1(point):
    x1 = point[0]
    x2 = point[1]
    x3 = point[2]
    x4 = point[3]
    x5 = point[4]
    x6 = point[5]
    x7 = point[6]
    x8 = point[7]

    r = (x1 - 30) ** 2 + (x2 - 18) ** 2 + (x3 - 48) ** 2 + (x4 - 30) ** 2 + (x5 - 68) ** 2 + (x6 - 62) ** 2 + (
            x7 - 90) ** 2 + (x8 - 76) ** 2
    return r


xopt0 = np.array([[30, 18, 48, 30, 68, 62, 90, 76]])
Ybest = 0


def out1(X):
    array = []
    for x in X:
        array.append(point_eval_out1(x))
    return np.array(array)


def my_pref(x1, x2, f=None):
    """
    Used to calculate π(x1,x2).
    :param x1: first vector of param. x1 in X
    :param x2:  second vector of param. x2 in X
    :param f: function of the evaluation
    :return: π(x1,x2). The return value is in {-1,0,1}
    """
    if f is not None:
        Y_pref = compute_preference(f, x1, x2)
        Y_pref.astype(int)
        print("best is :  %20s, next is: %20s, f(best): %4.4f, f(next): %4.4f, user's input: %d" % (
            str(x1), str(x2), f(x1), f(x2), Y_pref))
        return Y_pref
    else:
        print("best : ", x1, " ,     x next : ", x2)
        Y_pref = input("1 if x_next is worse than best so far, -1 otherwise")
        return np.array([int(Y_pref)])


def g(xt, Y, Models):
    return Y[:, 0]


bound1_1 = 20  # very slow
bound1_2 = 40
bound2_1 = 10  # vs1
bound2_2 = 26
bound3_1 = 36  # vs2
bound3_2 = 60
bound4_1 = 20  # vm1
bound4_2 = 40
bound5_1 = 60  # vm2
bound5_2 = 90
bound6_1 = 50  # vf1
bound6_2 = 70
bound7_1 = 70  # vf2
bound7_2 = 100
bound8_1 = 10  # acceleration
bound8_2 = 80
delta = 2

# Specify type and domain of the input variables
fvarsX = [
    {'name': 'x1', 'type': 'discrete', 'domain': np.around(np.arange(bound1_1, bound1_2 + 0.01, delta), 10)},
    {'name': 'x2', 'type': 'discrete', 'domain': np.around(np.arange(bound2_1, bound2_2 + 0.01, delta), 10)},
    {'name': 'x3', 'type': 'discrete', 'domain': np.around(np.arange(bound3_1, bound3_2 + 0.01, delta), 10)},
    {'name': 'x4', 'type': 'discrete', 'domain': np.around(np.arange(bound4_1, bound4_2 + 0.01, delta), 10)},
    {'name': 'x5', 'type': 'discrete', 'domain': np.around(np.arange(bound5_1, bound5_2 + 0.01, delta), 10)},
    {'name': 'x6', 'type': 'discrete', 'domain': np.around(np.arange(bound6_1, bound6_2 + 0.01, delta), 10)},
    {'name': 'x7', 'type': 'discrete', 'domain': np.around(np.arange(bound7_1, bound7_2 + 0.01, delta), 10)},
    {'name': 'x8', 'type': 'discrete', 'domain': np.around(np.arange(bound8_1, bound8_2 + 0.01, delta), 10)}
]

bb = 100

fvarsY = [{'name': 'out1', 'type': 'continuous', 'domain': ()}]

n = len(fvarsX)

init_samples = 6  # Number of initial samples

objectives = [out1]
objectivesFunctions = [point_eval_out1]

my_problem = GL.GLISpFinale(fvarsX, fvarsY, my_pref, objectives=objectives,
                            objectivesFunctions=objectivesFunctions,
                            acquisition_optimizer_type='lbfgsb',
                            kfold=3,
                            delta=1.5,
                            objective=(10, 1),
                            batch_size=1,
                            init_n_samples=init_samples,
                            save_experiment=True,
                            plot=False,
                            # plot=True,
                            title="PoliMi",
                            save=True,
                            g=g,
                            # theta=2,
                            # sigma=1 / 5,
                            # lam=0,
                            load_experiment_foldername='experimentsFolder/REDO',
                            name_opt='ENDGAME15', plotAcquisition=True)

exploration = True
x_next = my_problem.run_optimization(exploration=exploration)
iterations = 12


# print("X next: ", x_next, " -> ", utils.math.normalize_X(my_problem.fvars_x, x_next))


def reorder(xd, a):
    ee2 = a.reshape(len(a), 1)
    fin2 = np.column_stack((xd, ee2))
    fin2 = fin2[fin2[:, 0].argsort(), :]
    return fin2[:, 0], fin2[:, 1]


xx = np.arange(0, iterations, 1)  # sono gli step
f_x_best = []  # sono le f(best)
f_x_next = []  # sono le f(next)

end_explore = 3


def getExploration():
    exit = False
    while not exit:
        i = int(input("1 for exploration, 0 for exploitation"))
        if i == 1:
            return True
        elif i == 0:
            return False
        else:
            exit = False


for i in range(iterations):
    eval = []
    for ind in range(len(my_problem.models)):
        i_best = my_problem.Y_ind_best[ind]
        eval.append(
            my_pref(
                my_problem.models[ind].X[i_best:i_best + 1, :],
                my_problem.X_next[0: 1, :][:, ind * my_problem.ratio:(ind + 1) * my_problem.ratio]
            )
        )
    if i > iterations - end_explore:
      exploration = False
    my_problem.add_evaluations(x_next, eval)
    x_next = my_problem.run_optimization(exploration=exploration)

my_problem.update()

utils.process.printVar(my_problem.X, iterations, init_samples, end_explore)

utils.process.printBestForEachModel(my_problem)

name = "plot/" + "poliMi" + str(iterations) + "iterations" + datetime.now().strftime("%Y%m%d-%H%M%S")

utils.process.saveAux(name, len(my_problem.X), xopt0, np.shape(my_problem.X)[1], init_samples, end_explore,
                      my_problem.X)

utils.process.plotResults(len(fvarsX), len(fvarsY), xx, f_x_best, Ybest, name, objectivesFunctions, f_x_next,
                          my_problem.X, xopt0, init_samples,
                          end_explore)

i_best = my_problem.Y_ind_best[0]
delta = np.linalg.norm(my_problem.models[0].X[i_best:i_best + 1, :] - xopt0)
perc = delta / (bb * 2) * 100

print(my_problem.X)

print("Distance between opt and model's opt ", delta, " error: ", round(perc, 3), " %")
