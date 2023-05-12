import GlispFinale    as GL
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

    r = (x1 + 54) ** 2 + (x2 - 90) ** 2 + (x3 - 2) ** 2 + (x4 + 45) ** 2 + (x5 - 34) ** 2 + (x6 - 23) ** 2 + (
            x7 + 1) ** 2 + (x8 + 30) ** 2
    return r






xopt0 = np.array([[-54, 90, 2, -45, 34, 23, -1, -30]])
Ybest = 0


def out1(X):
    array = []
    for x in X:
        array.append(point_eval_out1(x))
    return np.array(array)


def my_pref(x1, x2, f=None):
    '''
    Used to calculate π(x1,x2).
    :param x1: first vector of param. x1 in X
    :param x2:  second vector of param. x2 in X
    :param f: function of the evaluation
    :return: π(x1,x2). The return value is in {-1,0,1}
    '''
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



bb = 100

fvarsX = [{'name': 'x1', 'type': 'discrete', 'domain': np.around(np.arange(-bb, bb+0.01, 2), 10)},
          {'name': 'x2', 'type': 'discrete', 'domain': np.around(np.arange(-bb, bb+0.01, 2), 10)},
          {'name': 'x3', 'type': 'discrete', 'domain': np.around(np.arange(-bb, bb+0.01, 2), 10)},
          {'name': 'x4', 'type': 'discrete', 'domain': np.around(np.arange(-bb, bb+0.01, 2), 10)},
          {'name': 'x5', 'type': 'discrete', 'domain': np.around(np.arange(-bb, bb+0.01, 2), 10)},
          {'name': 'x6', 'type': 'discrete', 'domain': np.around(np.arange(-bb, bb+0.01, 2), 10)},
          {'name': 'x7', 'type': 'discrete', 'domain': np.around(np.arange(-bb, bb+0.01, 2), 10)},
          {'name': 'x8', 'type': 'discrete', 'domain': np.around(np.arange(-bb, bb+0.01, 2), 10)}
          ]

fvarsY = [{'name': 'out1', 'type': 'continuous', 'domain': ()}]

n = len(fvarsX)

init_samples = 5  # Number of initial samples

objectives = [out1]
objectivesFunctions = [point_eval_out1]

my_problem = GL.GLISpFinale(fvarsX, fvarsY, my_pref, objectives=objectives,
                            objectivesFunctions=objectivesFunctions,
                            acquisition_optimizer_type='lbfgsb',
                            kfold=3, delta=0.5,
                            objective=(10, 1),
                            batch_size=1, init_n_samples=init_samples,
                            save_experiment=True, plot=False,
                            title="Bemporad Function", save=True,
                            g=g,
                            # theta=2, sigma=1 / 5, lam=0,
                            #load_experiment_foldername='experimentsFolder/Experiment_poliMi3_20211130-150857',
                            max_bound=bb,
                            name_opt='poliMi3', plotAcquisition=False)

exploration = True
x_next = my_problem.run_optimization(exploration=exploration)
iterations = 35
print("X next: ", x_next, " -> ", utils.math.normalize_X(my_problem.fvars_x, x_next))


def reorder(xd, a):
    ee2 = a.reshape(len(a), 1)
    fin2 = np.column_stack((xd, ee2))
    fin2 = fin2[fin2[:, 0].argsort(), :]
    return fin2[:, 0], fin2[:, 1]


xx = np.arange(0, iterations, 1)  # sono gli step
f_x_best = []  # sono le f(best)
f_x_next = []  # sono le f(next)

end_explore = 3
for i in range(iterations):
    # t.sendIt((x_next[0]).tolist())
    eval = []
    y_sum = 0
    y_sum_opt = 0
    print(i)
    for ind in range(len(my_problem.models)):
        i_best = my_problem.Y_ind_best[ind]
        eval.append(
            my_pref(
                my_problem.models[ind].X[i_best:i_best + 1, :],

                my_problem.X_next[0: 1, :][:, ind * my_problem.ratio:(ind + 1) * my_problem.ratio],

                f=objectives[ind]
            )
        )  # for all y there will be an evaluation
        y_sum += objectives[ind](x_next)
        y_sum_opt += objectives[ind](my_problem.models[ind].X[i_best:i_best + 1, :])
    f_x_next.append(y_sum)
    f_x_best.append(y_sum_opt)

    if i > iterations - end_explore:
        exploration = False
    # if i % 2:
    #     exploration = False
    # else:
    #     exploration = True

    my_problem.add_evaluations(x_next, eval)
    x_next = my_problem.run_optimization(exploration=exploration)

my_problem.update()

utils.process.printVar(my_problem.X, iterations, init_samples, end_explore)

utils.process.printBestForEachModel(my_problem)

name = "plot/" + "Ex_2X1Y" + str(iterations) + "iterations" + datetime.now().strftime("%Y%m%d-%H%M%S")

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
