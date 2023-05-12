import numpy as np
import utils.math
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.mplot3d import proj3d
import os
import inspect


def visualize3DData(X, init_samples, end_explore, name=None):
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    adj = 1.5
    a = X[0:1, 0]
    b = X[0:1, 1]
    c = X[0:1, 2]
    ax.scatter(a, b, c, depthshade=False, picker=True, color="green", label="optimum", s=10 + adj)

    a = X[1:init_samples + 1, 0]
    b = X[1:init_samples + 1, 1]
    c = X[1:init_samples + 1, 2]

    ax.scatter(a, b, c, depthshade=False, picker=True, color="blue", label="initial points", s=2 + adj)

    a = X[init_samples + 1:len(X) - end_explore, 0]
    b = X[init_samples + 1:len(X) - end_explore, 1]
    c = X[init_samples + 1:len(X) - end_explore, 2]
    #
    ax.scatter(a, b, c, depthshade=False, picker=True, color="red", label="exploration", s=2 + adj)

    a = X[len(X) - end_explore:, 0]
    b = X[len(X) - end_explore:, 1]
    c = X[len(X) - end_explore:, 2]
    ax.scatter(a, b, c, depthshade=False, picker=True, color="black", label="exploitation", s=3.5 + adj)

    def distance(point, event):
        """Return distance between mouse position and given data point

        Args:
            point (np.array): np.array of shape (3,), with x,y,z in data coords
            event (MouseEvent): mouse event (which contains mouse position in .x and .xdata)
        Returns:
            distance (np.float64): distance (in screen coords) between mouse pos and data point
        """
        assert point.shape == (3,), "distance: point.shape is wrong: %s, must be (3,)" % point.shape

        # Project 3d data space to 2d data space
        x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], plt.gca().get_proj())
        # Convert 2d data space to 2d screen space
        x3, y3 = ax.transData.transform((x2, y2))

        return np.sqrt((x3 - event.x) ** 2 + (y3 - event.y) ** 2)

    def calcClosestDatapoint(X, event):
        """"Calculate which data point is closest to the mouse position.

        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            event (MouseEvent) - mouse event (containing mouse position)
        Returns:
            smallestIndex (int) - the index (into the array of points X) of the element closest to the mouse position
        """
        distances = [distance(X[i, 0:3], event) for i in range(X.shape[0])]
        return np.argmin(distances)

    def annotatePlot(X, index):
        """Create popover label in 3d chart

        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            index (int) - index (into points array X) of item which should be printed
        Returns:
            None
        """
        # If we have previously displayed another label, remove it first
        if hasattr(annotatePlot, 'label'):
            annotatePlot.label.remove()
        # Get data point from array of points X, at position index
        x2, y2, _ = proj3d.proj_transform(X[index, 0], X[index, 1], X[index, 2], ax.get_proj())
        s = "idx: " + str(index) + "\n" + "x: " + str(X[index, 0]) + "\n" + "y: " + str(
            X[index, 1]) + "\n" + "z:" + str(
            X[index, 2])
        annotatePlot.label = plt.annotate("%s" % s,
                                          xy=(x2, y2), xytext=(-20, 20), textcoords='offset points', ha='right',
                                          va='bottom',
                                          bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                                          arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        fig.canvas.draw()

    def onMouseMotion(event):
        """Event that is triggered when mouse is moved. Shows text annotation over data point closest to mouse."""
        closestIndex = calcClosestDatapoint(X, event)
        annotatePlot(X, closestIndex)

    fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)  # on mouse motion
    plt.suptitle(
        str(init_samples) + " inital points, " + str(
            len(X) - init_samples - end_explore - 1) + " exploration points, " + str(
            end_explore) + " exploitation points.")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=4, fancybox=True, shadow=True)
    # save(plt, "name", False, "testedPoints", None)

    plt.savefig(name + "/" + "testedPoints" + ".pdf")
    plt.show()


def _plot(a, xd, z, x, plot, title, nameOpt, n_iteration, RBF_model, X_next, maxBound, my_fun_easy, save):
    if len(x) > 100 and plot:
        print("****** PLOTTING STUFF ******")
        fig, axs = plt.subplots(3, 1, figsize=(15, 15))
        plt.grid()
        ee = z.reshape(len(z), 1)
        fin = np.column_stack((xd, ee))
        fin = fin[fin[:, 0].argsort(), :]
        yMax = 2.5
        # axs.axis([-3, 3, 0, yMax])
        if title is not None:
            fig.suptitle(
                'expermient name: ' + nameOpt + ', iteration ' + str(n_iteration) + ", " + title,
                fontsize=14,
                fontweight='bold')
        else:
            fig.suptitle('expermient name: ' + nameOpt + ', iteration ' + str(n_iteration), fontsize=14,
                         fontweight='bold')
        axs[0].plot(fin[:, 0], fin[:, 1], label="z(x)")  # ".b",
        # axs[0].set_ylim([0, 0.004])

        axs[0].set_title("z(x): acquisition function without exploration")
        axs[0].plot(RBF_model.X, np.zeros(len(RBF_model.X)), "ro")
        axs[0].grid()
        if X_next is not None:
            axs[0].plot(X_next, [0], ".b")
        plt.grid()
        ee2 = a.reshape(len(a), 1)
        fin2 = np.column_stack((xd, ee2))
        fin2 = fin2[fin2[:, 0].argsort(), :]
        index = np.argmin(fin2[:, 1])
        st = str(fin2[index, 0]) + "  " + str(fin2[index, 1])
        axs[1].plot(fin2[index, 0], fin2[index, 1], "ro")
        axs[1].set_title("a(x): acquisition function with exploration, x_next: " + str(X_next) + " --> " + st)
        axs[1].plot(fin2[:, 0], fin2[:, 1], label='a(x)')  # ".b",
        axs[1].grid()
        axs[2].set_title("f(x) vs f_hat(x)")
        index = 0
        for xc in RBF_model.X:
            axs[2].axvline(x=xc)
            axs[2].text(xc, -0.4, "x" + str(index), fontsize=15)
            index += 1
        spazio = np.linspace(- maxBound, maxBound, 1000)
        axs[2].plot(spazio, utils.math.scaleValues(RBF_model.predict(spazio), 2.5),
                    label='f_hat(x), surrogate')
        axs[2].plot(spazio, my_fun_easy(spazio), label="f(x), true function")
        axs[2].grid()
        plt.legend()
        if save:
            plt.savefig("glispMedia/acquisition/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        else:
            plt.show()


def _plot2(a, xd, z, x, plot, title, nameOpt, n_iteration, RBF_model, counter, X_next, maxBound, X, obj, save=False):
    if len(x) > 100 and plot:
        print("****** PLOTTING STUFF ******")
        fig, axs = plt.subplots(3, 1, figsize=(15, 15))
        plt.grid()
        ee = z.reshape(len(z), 1)
        fin = np.column_stack((xd, ee))
        fin = fin[fin[:, 0].argsort(), :]
        yMax = 2.5
        # axs.axis([-3, 3, 0, yMax])
        if title is not None:
            fig.suptitle(
                'expermient name: ' + nameOpt + ', iteration ' + str(n_iteration) + ", Model # " + str(
                    counter) + "  " + title, fontsize=14,
                fontweight='bold')
        else:
            fig.suptitle(
                'expermient name: ' + nameOpt + ', iteration ' + str(n_iteration) + ", Model # " + str(counter),
                fontsize=14, fontweight='bold')
        # axs[0].plot(fin[:, 0], fin[:, 1], label="z(x)")  # ".b",
        axs[0].plot(xd, z, "ro")  # ".b",
        # axs[0].set_ylim([0, 0.004])

        axs[0].set_title("z(x): acquisition function without exploration")
        axs[0].plot(RBF_model.X, np.zeros(len(RBF_model.X)), "ro")

        axs[0].grid()
        if X_next is not None:
            axs[0].plot(X_next, [0], ".b")
        plt.grid()
        ee2 = a.reshape(len(a), 1)
        fin2 = np.column_stack((xd, ee2))
        fin2 = fin2[fin2[:, 0].argsort(), :]
        index = np.argmin(fin2[:, 1])
        # axs[1].plot(fin2[index, 0], fin2[index, 1], "ro")
        axs[1].plot(xd, a, "ro")
        st = str(fin2[index, 0]) + "  " + str(fin2[index, 1])
        axs[1].set_title("a(x): acquisition function with exploration, x_next: " + str(X_next) + " --> " + st)
        axs[1].plot(fin2[:, 0], fin2[:, 1], label='a(x)')  # ".b",
        axs[1].grid()
        axs[2].set_title("f(x) vs f_hat(x)")
        index = 0
        for xc in X:
            axs[2].axvline(x=xc)
            axs[2].text(xc, -0.4, "x" + str(index), fontsize=15)
            index += 1
        spazio = np.linspace(- maxBound, maxBound, 1000)
        axs[2].plot(spazio, utils.math.scaleValues(RBF_model.predict(spazio), 2.5),
                    label='f_hat(x), surrogate')
        #        axs[2].plot(spazio, obj(spazio), label="f(x), true function")
        axs[2].grid()
        plt.legend()
        if save:
            plt.savefig("glispMedia/acquisition/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        else:
            plt.show()


def printingStuff(RBF_model):
    print("Value in RBF_model.x:")
    print(RBF_model.X)
    print("**** Fine ****")
    '''
            print('pred class 0:')
            print(self.RBF_model.GP_loss[0].predict_proba(self.RBF_model.X)[:,0:1])
            print('pred class 1:')
            print(self.RBF_model.GP_loss[0].predict_proba(self.RBF_model.X)[:,0:1])
            input("Fine predictions")
            '''
    # constraints
    # self.acquisition.set_constraints(self.constraints)


def testingSurrogateAndAcquisition(testing, RBF_model, IDW_acquisition):
    if testing:
        xopt = np.array([[5., 5.]])
        xn = utils.math.normalize_X(RBF_model.fvars_x, xopt)
        facq = IDW_acquisition(xn)
        f_hat = RBF_model.predict(xopt)
        # print('Value of acquisition at the optizer: %2.2f; and surrogate: %2.2f' % (facq, f_hat))


def printInfoExperiment(delta, ind_best, RBF_model):
    print('delta: ', delta)
    print('self.ind_best: ', ind_best)
    print('n exp: ', RBF_model.X.shape[0])
    np.set_printoptions(suppress=True)
    print('self.RBF_model.X[best]: ', RBF_model.X[ind_best, :])
    # np.savetxt(sys.stdout, self.RBF_model.X[self.ind_best, :], '%5.2f')


def save(plti, name, var, nomeFile, objectivesFunctions):
    folder_exists = os.path.exists(name)
    if not folder_exists:
        os.makedirs(name)
    plti.savefig(name + "/" + nomeFile + ".pdf")

    if var:
        lines = ''
        con = 0
        for e in objectivesFunctions:
            lines += "FUNCTION EVAL #" + str(con) + ": \n"
            con += 1
            lines += inspect.getsource(e) + "\n"
        s = name + '/info.txt'
        text_file = open(s, "w")
        text_file.write(lines)
        text_file.close()


from pathlib import Path

def saveAux(name, len, optimal, dim, init_sample, endExplore, X):
    folder_exists = os.path.exists(name)

    if not folder_exists:
        os.makedirs(name)

    lines = ''
    lines += "Len:" + str(len) + "\n"

    lines += "Dim:" + str(dim) + "\n"
    lines += "InitSamples:" + str(init_sample) + "\n"
    lines += "EndExplore:" + str(endExplore) + "\n"
    s = name + '/aux.txt'
    #text_file = open(s, "w")
    #text_file.write(lines)
    #text_file.close()

    s = name + '/optimal.csv'
    np.savetxt(s, optimal, delimiter=',')
    s = name + '/X.csv'
    np.savetxt(s, X, delimiter=',')


def plotResults(dimensionX, dimensionY, xx, yy, Ybest, name, objectivesFunctions, fyy, X, xopt0, init_samples,
                end_explore):
    if dimensionY == 1:
        mmm = 50
        plt.title("Achievement of the optimum")
        plt.plot(xx, yy)
        appoggio = np.ones(len(xx)) * Ybest
        plt.plot(xx, appoggio)
        plt.ylim((Ybest - 0.5), mmm)  # (np.min(yy) + np.min(yy)*0.2))
        # plt.ylim((Ybest - 0.5), (Ybest + 5))
        plt.xlabel("iterations")
        plt.ylabel("f(" + r'$x_{best}$)')
        plt.grid()
        save(plt, name, True, "ottimoScalato", objectivesFunctions)
        plt.show()

        # plt.title("Achievement of the optimum")
        # plt.plot(xx, yy)
        # appoggio = np.ones(len(xx)) * Ybest
        # plt.plot(xx, appoggio)
        # plt.grid()
        # plt.xlabel("iterations")
        # plt.ylabel("f(" + r'$x_{best}$)')
        # yy=np.nan_to_num(yy, neginf=0, posinf=100000)
        # plt.ylim(round(Ybest - 0.5), (np.max(yy) + 2))
        # save(plt, name, False, "ottimoNonScalato", objectivesFunctions)
        # plt.show()

        plt.title("f(x_next)")
        plt.plot(xx, fyy)
        appoggio = np.ones(len(xx)) * Ybest
        plt.plot(xx, appoggio)
        plt.grid()
        plt.xlabel("iterations")
        plt.ylabel("f(" + r'$x_{next}$' + ")")
        plt.ylim((Ybest - 0.5), mmm)  # (np.min(fyy) + np.min(fyy)*0.2))
        save(plt, name, False, "f(xNext)Zoom", objectivesFunctions)
        plt.show()

        # plt.title("f(x_next)")
        # plt.plot(xx, fyy)
        # appoggio = np.ones(len(xx)) * Ybest
        # plt.plot(xx, appoggio)
        # plt.grid()
        # plt.xlabel("iterations")
        # plt.ylabel("f(" + r'$x_{next}$' + ")")
        # fyy = np.nan_to_num(fyy, neginf=0, posinf=100000)
        # plt.ylim((Ybest - 0.5), (np.max(fyy) + 0.5))
        # save(plt, name, False, "f(xNext)", objectivesFunctions)
        # plt.show()
    else:
        plt.title("f(x_next)")
        plt.plot(xx, fyy)
        appoggio = np.ones(len(xx)) * Ybest
        plt.plot(xx, appoggio)
        plt.ylim((Ybest - 0.5), (Ybest + 5))
        plt.grid()
        save(plt, name, True, "f(xNext)scalato", objectivesFunctions)
        plt.show()

        plt.title("f(x_next)")
        plt.plot(xx, fyy)
        appoggio = np.ones(len(xx)) * Ybest
        plt.plot(xx, appoggio)
        plt.grid()
        plt.ylim((Ybest - 0.5), (np.max(fyy) + 0.5))
        save(plt, name, True, "f(xNext)NonScalato", objectivesFunctions)
        plt.show()
    if dimensionX == 1:
        plt.figure(figsize=(7, 6), dpi=300)
        plt.suptitle(
            str(init_samples) + " inital points, " + str(
                len(X) - init_samples - end_explore) + " exploration points, " + str(
                end_explore) + " exploitation points.")
        for er in xopt0:
            a = er[0]
            b = 0
            plt.plot(a, b, 'x', c='green', label="best", markersize=18)

        x = X
        a = x[0:init_samples, 0]
        b = np.zeros_like(a)
        plt.plot(a, b, 'o', c='blue', label="initial points", markersize=2.5)
        a = x[init_samples:len(x) - end_explore, 0]
        b = np.zeros_like(a)

        plt.plot(a, b, 'o', c='red', label="exploration", markersize=2.5)

        a = x[len(x) - end_explore:, 0]
        b = np.zeros_like(a)
        plt.plot(a, b, 'o', c='black', label="exploitation", markersize=2.5)

        text = False
        if text:
            co = 0
            for w in x:
                plt.text(w[0], 0, str(co), fontsize=8)
                co += 1

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09),
                   ncol=4, fancybox=True, shadow=True)
        plt.grid()
        plt.xlabel("X")
        plt.ylabel("Y")
        save(plt, name, False, "testedPoints", objectivesFunctions)
        plt.show()
    elif dimensionX == 2:
        plt.figure(figsize=(7, 6), dpi=300)
        plt.suptitle(
            str(init_samples) + " inital points, " + str(
                len(X) - init_samples - end_explore) + " exploration points, " + str(
                end_explore) + " exploitation points.")
        for er in xopt0:
            a = er[0]
            b = er[1]
            plt.plot(a, b, 'x', c='green', label="best", markersize=18)

        x = X
        a = x[0:init_samples, 0]
        b = x[0:init_samples, 1]
        plt.plot(a, b, 'o', c='blue', label="initial points", markersize=2.5)
        a = x[init_samples:len(x) - end_explore, 0]
        b = x[init_samples:len(x) - end_explore, 1]

        plt.plot(a, b, 'o', c='red', label="exploration", markersize=2.5)

        a = x[len(x) - end_explore:, 0]
        b = x[len(x) - end_explore:, 1]
        plt.plot(a, b, 'o', c='black', label="exploitation", markersize=2.5)

        text = False
        if text:
            co = 0
            for w in x:
                plt.text(w[0], w[1], str(co), fontsize=8)
                co += 1

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09),
                   ncol=4, fancybox=True, shadow=True)
        plt.grid()
        plt.xlabel("X")
        plt.ylabel("Y")
        save(plt, name, False, "testedPoints", objectivesFunctions)
        plt.show()
    elif dimensionX == 3:
        X = np.concatenate((xopt0, X))
        visualize3DData(X, init_samples, end_explore, name)


def printVar(X, iterations, init_samples, end_explore):
    print("X value:")
    index = 0
    index2 = 0
    if np.shape(X)[1] == 1:
        for x in X:
            if index < init_samples:
                print("%s     %5.5f" % ("initial", x[0]))
            else:
                if index2 > iterations - end_explore:
                    print("%8d*:   %5.5f" % (index2, x[0]))
                else:
                    print("%8d :   %5.5f" % (index2, x[0]))
                index2 += 1
            index += 1
    elif np.shape(X)[1] == 2:
        for x in X:
            if index < init_samples:
                print("%s     %5.5f ,   %5.5f" % ("initial", x[0], x[1]))
            else:
                if index2 > iterations - end_explore:
                    print("%8d*:   %5.5f ,   %5.5f" % (index2, x[0], x[1]))
                else:
                    print("%8d :   %5.5f ,   %5.5f" % (index2, x[0], x[1]))
                index2 += 1
            index += 1
    elif np.shape(X)[1] == 3:
        for x in X:
            if index < init_samples:
                print("%s     %5.5f ,   %5.5f,   %5.5f" % ("initial", x[0], x[1], x[2]))
            else:
                if index2 > iterations - end_explore:
                    print("%8d*:   %5.5f ,   %5.5f,   %5.5f" % (index2, x[0], x[1], x[2]))
                else:
                    print("%8d :   %5.5f ,   %5.5f,   %5.5f" % (index2, x[0], x[1], x[2]))
                index2 += 1
            index += 1
    elif np.shape(X)[1] == 4:
        for x in X:
            if index < init_samples:
                print("%s     %5.5f ,   %5.5f,   %5.5f,   %5.5f" % ("initial", x[0], x[1], x[2], x[3]))
            else:
                if index2 > iterations - end_explore:
                    print("%8d*:   %5.5f ,   %5.5f,   %5.5f,   %5.5f" % (index2, x[0], x[1], x[2], x[3]))
                else:
                    print("%8d :   %5.5f ,   %5.5f,   %5.5f,   %5.5f" % (index2, x[0], x[1], x[2], x[3]))
                index2 += 1
            index += 1
    elif np.shape(X)[1] == 6:
        for x in X:
            if index < init_samples:
                print("%s     %5.5f ,   %5.5f,   %5.5f,   %5.5f,   %5.5f,   %5.5f" % (
                    "initial", x[0], x[1], x[2], x[3], x[4], x[5]))
            else:
                if index2 > iterations - end_explore:
                    print("%8d*:   %5.5f ,   %5.5f,   %5.5f,   %5.5f,   %5.5f,   %5.5f" % (
                        index2, x[0], x[1], x[2], x[3], x[4], x[5]))
                else:
                    print("%8d :   %5.5f ,   %5.5f,   %5.5f,   %5.5f,   %5.5f,   %5.5f" % (
                        index2, x[0], x[1], x[2], x[3], x[4], x[5]))
                index2 += 1
            index += 1


def printBestForEachModel(my_problem):
    for ind in range(len(my_problem.models)):
        i_best = my_problem.Y_ind_best[ind]
        print("Best: ", str(my_problem.models[ind].X[i_best:i_best + 1, :]))


def retBest(my_problem):
    arr = []
    for ind in range(len(my_problem.models)):
        i_best = my_problem.Y_ind_best[ind]
        arr.append(my_problem.models[ind].X[i_best:i_best + 1, :])
    return arr
