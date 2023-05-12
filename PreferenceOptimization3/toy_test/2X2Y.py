import GlispFinale    as GL
import numpy as np
from utils.preference_functions import compute_preference
# from rosFolder import talkerPreferencesMultiArray as t
from datetime import datetime
import utils.process


def d(x):
    x1 = x[0]
    x2 = x[1]
    return point_eval_out1(x1) + point_eval_out2(x2)


def point_eval_out1(point):
    x = point
    # y = point[1]
    # r = (4 - 2.1 * x ** 2 + x ** 4 / 3) * x ** 2 + x
    r = (x - 34) ** 2
    # xopts = [[0.0898, -0.0898],[-0.7126, 0.7126]]  # unconstrained optimizers, one per column
    # fopts = -1.03  # unconstrained optimum
    return r


def point_eval_out2(point):
    y = point
    r = (y + 77) ** 2
    # xopts = [0,0]  # unconstrained optimizers, one per column
    # fopts = 0 # unconstrained optimum
    # this two function, combined, have opt point and opt y at
    # xopt0 = np.array([[-0.061, 0.61], [0.061, -0.61]])
    # Ybest = -0.58
    return r


# this two function, combined, have opt point and opt y at
xopt0 = np.array([[34, -77]])
Ybest = 0


def out1(X):
    array = []
    for x in X:
        array.append(point_eval_out1(x))
    return np.array(array)


def out2(X):
    array = []
    for x in X:
        array.append(point_eval_out2(x))
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
    return Y[:, 0] + Y[:, 1]


bb = 2000

fvarsX = [
    {'name': 'x1', 'type': 'continuous', 'domain': (-bb, bb)},
    {'name': 'x2', 'type': 'continuous', 'domain': (-bb, bb)}
]

fvarsY = [
    {'name': 'out1', 'type': 'continuous', 'domain': ()},
    {'name': 'out2', 'type': 'continuous', 'domain': ()},
]

n = len(fvarsX)

init_samples = 5  # Number of initial samples

objectives = [out1, out2]
objectivesFunctions = [point_eval_out1, point_eval_out2]

my_problem = GL.GLISpFinale(fvarsX, fvarsY, my_pref, objectives=objectives,
                            objectivesFunctions=objectivesFunctions,
                            acquisition_optimizer_type='lbfgsb',
                            kfold=3, delta=5,
                            objective=(10, 1),
                            batch_size=1,
                            save_experiment=True,
                            g=g,
                            save=True,
                            # theta=2, sigma=1 / 5, lam=2,
                            # load_experiment_foldername='experimentsFolder/Experiment_2x2y_20210803-114556',
                            max_bound=bb,
                            name_opt='2x2y', plotAcquisition=False)

exploration = True
x_next = my_problem.run_optimization(exploration=exploration)
iterations = 35

xx = np.arange(0, iterations, 1)  # sono gli step

f_x_next = []  # sono le f(best) raggiungimento ottimo

end_explore = 3
for i in range(iterations):
    eval = []
    print(i)
    for ind in range(len(my_problem.models)):
        i_best = my_problem.Y_ind_best[ind]
        eval.append(
            my_pref(
                my_problem.models[ind].X[i_best:i_best + 1, :],

                my_problem.X_next[0: 1, :][:, ind * my_problem.ratio:(ind + 1) * my_problem.ratio],

                f=objectives[ind])
        )  # for all y there will be an evaluation
    f_x_next.append(d(x_next[0]))

    if i > iterations - end_explore:
        exploration = False

    # invece di fare add di xnext faccio add del punto di scipy
    #
    my_problem.add_evaluations(x_next, eval)

    x_next = my_problem.run_optimization(exploration=exploration)
    print("X NEXT è  ", x_next)

utils.process.printVar(my_problem.X, iterations, init_samples, end_explore)

utils.process.printBestForEachModel(my_problem)

name = "plot/" + "Ex_2X2Y" + str(iterations) + "iterations" + datetime.now().strftime("%Y%m%d-%H%M%S")

utils.process.saveAux(name, len(my_problem.X), xopt0, np.shape(my_problem.X)[1], init_samples, end_explore,
                      my_problem.X)

utils.process.plotResults(len(fvarsX), len(fvarsY), xx, None, Ybest, name, objectivesFunctions, f_x_next, my_problem.X,
                          xopt0, init_samples,
                          end_explore)

bl = np.zeros_like(xopt0, dtype=float)
(N, n) = np.shape(bl)
contatore = 0
for ind in range(len(my_problem.models)):
    i_best = my_problem.Y_ind_best[ind]
    for el in my_problem.models[ind].X[i_best:i_best + 1, :]:
        bl[0, contatore] = el
        contatore += 1
print(bl)

delta = np.linalg.norm(bl - xopt0)
perc = delta / (bb * 2) * 100
print("Distance between opt and model's opt ", delta, " error: ", round(perc, 3), " %")
