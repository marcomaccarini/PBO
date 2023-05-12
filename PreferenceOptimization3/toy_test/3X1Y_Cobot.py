import PreferenceOptimization3.GlispFinale as GL
import numpy as np
from PreferenceOptimization3.utils.preference_functions import compute_preference
# from rosFolder import talkerPreferencesMultiArray as t
from datetime import datetime

import numpy.linalg
from PreferenceOptimization3.utils import process

def point_eval_out1(point):
    x1 = point[0]
    x2 = point[1]
    x3 = point[2]

    r = (x1 - 550) ** 2 + (x2 - 1) ** 2 + (x3 - 1000) ** 2
    return r


xopt0 = np.array([[550, 1, 1000]])
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
    # np.set_printoptions(suppress=True)
    if f is not None:
        Y_pref = compute_preference(f, x1, x2)
        Y_pref.astype(int)

        print("best is :  %20s, next is: %20s, f(best): %4.4f, f(next): %4.4f, user's input: %d" % (
            str(x1), str(x2), f(x1), f(x2), Y_pref))
        return Y_pref
    else:
        print("1: ", x2[0][0])
        print("2: ", x2[0][1])
        print("3: ", x2[0][2])

        print("best : ", x1, " ,     x next : ", x2)
        Y_pref = input("1 if x_next is worse than best so far, -1 otherwise ")
        return np.array([int(Y_pref)])


def g(xt, Y, Models):
    return Y[:, 0]


bound1_1 = 250  # very slow
bound1_2 = 1500
bound2_1 = 0.4  # vs1
bound2_2 = 2
bound3_1 = 0.001  # 50  # vs2
bound3_2 = 10  # 2000

# Specify type and domain of the input variables
fvarsX = [
    {'name': 'x1', 'type': 'discrete', 'domain': np.arange(-1000, 1000, 50)},
          {'name': 'x2', 'type': 'discrete', 'domain': np.arange(-1000, 1000, 1)},
          {'name': 'x3', 'type': 'discrete', 'domain': np.arange(-1000, 1500, 50)},

          ]

bb = 1000

fvarsY = [{'name': 'out1', 'type': 'continuous', 'domain': ()}]

n = len(fvarsX)

init_samples = 6  # Number of initial samples

objectives = [out1]
objectivesFunctions = [point_eval_out1]
np.set_printoptions(suppress=True)
my_problem = GL.GLISpFinale(fvarsX, fvarsY, my_pref, objectives=objectives,
                            objectivesFunctions=objectivesFunctions,
                            acquisition_optimizer_type='lbfgsb',
                            kfold=3, delta=2.5,
                            objective=(10, 1),
                            batch_size=1, init_n_samples=init_samples,
                            save_experiment=True, plot=False,
                            title="PoliMi", save=True,
                            g=g,
                            # theta=2, sigma=1 / 5, lam=0,
                            load_experiment_foldername='experimentsFolder/real',
                            name_opt='provaReale', plotAcquisition=False)

exploration = True
x_next = my_problem.run_optimization(exploration=exploration)
iterations = 30


# 250 0.9 3

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
        i = int(input("1 for exploration, 0 for exploitation: "))
        if i == 1:
            return True
        elif i == 0:
            return False
        else:
            exit = False


exploration = True
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

                my_problem.X_next[0: 1, :][:, ind * my_problem.ratio:(ind + 1) * my_problem.ratio]  ,

                f=objectives[ind]
            )
        )  # for all y there will be an evaluation
        y_sum += 0  # objectives[ind](x_next)
        y_sum_opt += 0  # objectives[ind](my_problem.models[ind].X[i_best:i_best + 1, :])
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

process.printVar(my_problem.X, iterations, init_samples, end_explore)

process.printBestForEachModel(my_problem)

name = "plot/" + "IDSIA" + str(iterations) + "iterations" + datetime.now().strftime("%Y%m%d-%H%M%S")

process.saveAux(name, len(my_problem.X), xopt0, np.shape(my_problem.X)[1], init_samples, end_explore,
                      my_problem.X)

# utils.process.plotResults(len(fvarsX), len(fvarsY), xx, f_x_best, Ybest, name, objectivesFunctions, f_x_next,
#                          my_problem.X, xopt0, init_samples,
#                          end_explore)

i_best = my_problem.Y_ind_best[0]
delta = np.linalg.norm(my_problem.models[0].X[i_best:i_best + 1, :] - xopt0)
perc = delta / (bb * 2) * 100

#print(my_problem.X)

print("Distance between opt and model's opt ", delta, " error: ", round(perc, 3), " %")
