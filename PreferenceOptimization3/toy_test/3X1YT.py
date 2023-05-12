import PreferenceOptimization3.GlispFinale as GL
import numpy as np
from PreferenceOptimization3.utils.preference_functions import compute_preference
# from rosFolder import talkerPreferencesMultiArray as t
from datetime import datetime
import PreferenceOptimization3.utils.process
import numpy.linalg


def point_eval_out1(point):
    x1 = point[0]
    x2 = point[1]
    x3 = point[2]
    r = (x1 + 350) ** 2 + (x2 - 1) ** 2 + (x3 - 1000) ** 2
    return r


xopt0 = [[-350, 1, 1000]]
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


bb = 1200

fvarsX = [{'name': 'stiffness', 'type': 'discrete', 'domain': np.arange(-bb, bb, 1)},
          {'name': 'damping', 'type': 'discrete', 'domain': np.arange(-bb, bb, 0.5)},
          {'name': 'kp', 'type': 'discrete', 'domain': np.arange(-bb, bb, 0.1)},
          ]
fvarsY = [
    {'name': 'out1', 'type': 'continuous', 'domain': ()}
]

n = len(fvarsX)

init_samples = 5  # Number of initial samples

objectives = [out1]
objectivesFunctions = [point_eval_out1]

my_problem = GL.GLISpFinale(fvarsX, fvarsY, my_pref, objectives=objectives,
                            objectivesFunctions=objectivesFunctions,
                            acquisition_optimizer_type='lbfgsb',
                            kfold=3, delta=2,
                            objective=(10, 1),
                            batch_size=1,
                            save_experiment=True, plot=True,
                            g=g,
                            # theta=2, sigma=1 / 5, lam=2,
                            #  load_experiment_foldername='experimentsFolder/Experiment_44',
                            max_bound=bb,
                            name_opt='Cobot', plotAcquisition=False)

exploration = True
x_next = my_problem.run_optimization(exploration=exploration)
iterations = 25
print("X next: ", x_next)

xx = np.arange(0, iterations, 1)  # sono gli step
f_x_best = []
f_x_next = []

end_explore = 5  # int(iterations * 0.1)
print("End explore: ", end_explore)
for i in range(iterations):
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
    my_problem.add_evaluations(x_next, eval)
    x_next = my_problem.run_optimization(exploration=exploration)
    # print(x_next)

utils.process.printVar(my_problem.X, iterations, init_samples, end_explore)

utils.process.printBestForEachModel(my_problem)

name = "plot/" + "Ex_3X1Y" + str(iterations) + "iterations" + datetime.now().strftime("%Y%m%d-%H%M%S")

utils.process.saveAux(name, len(my_problem.X), xopt0, np.shape(my_problem.X)[1], init_samples, end_explore,
                      my_problem.X)

utils.process.plotResults(len(fvarsX), len(fvarsY), xx, f_x_best, Ybest, name, objectivesFunctions, f_x_next,
                          my_problem.X, xopt0, init_samples,
                          end_explore)

i_best = my_problem.Y_ind_best[0]
delta = numpy.linalg.norm(my_problem.models[0].X[i_best:i_best + 1, :] - xopt0)
print("Distance between opt and model's opt ", delta)
