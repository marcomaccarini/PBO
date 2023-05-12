import pandas as pd

import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import proj3d


def visualize3DData(X):
    """Visualize data in 3d plot with popover next to mouse position.

    Args:
        X (np.array) - array of points, of shape (numPoints, 3)
    Returns:
        None
    """
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
        str(init_samples) + "inital -- " + str(len(X) - end_explore - init_samples) + " exploitation --" + str(
            end_explore) + " exploitation")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=4, fancybox=True, shadow=True)
    plt.show()


if __name__ == '__main__':
    name = "plot/3X1Y30iterations20210705-143204"

    file1 = open(name + "/aux.txt", 'r')
    Lines = file1.readlines()

    count = 0
    leng = dim = init_samples = end_explore = 0
    # Strips the newline character
    for line in Lines:
        count += 1
        a = str(line.strip()).split(":")

        if a[0] == "Len":
            leng = int(a[1])
        elif a[0] == "Dim":
            dim = int(a[1])
        elif a[0] == "InitSamples":
            init_samples = int(a[1])
        elif a[0] == "EndExplore":
            end_explore = int(a[1])
    s_X = name + "/optimal.csv"
    xopt0 = pd.read_csv(s_X, header=None).values

    print(leng)
    print(dim)
    print(init_samples)
    print(end_explore)
    print(xopt0)

    s_X = name + "/X.csv"
    X = pd.read_csv(s_X, header=None).values

    if dim == 2:
        plt.figure(figsize=(7, 6), dpi=300)
        plt.suptitle("punti testati")
        plt.suptitle(
            str(init_samples) + "inital -- " + str((len(X) - init_samples - end_explore)) + " exploration --" + str(
                end_explore) + " exploitation")
        a = xopt0[:, 0]
        b = xopt0[:, 1]
        plt.plot(a, b, 's', c='green', label="best", markersize=4)

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
        plt.show()
    elif dim == 3:
        X = X
        X = np.concatenate((xopt0, X))
        print(X)
        visualize3DData(X)
