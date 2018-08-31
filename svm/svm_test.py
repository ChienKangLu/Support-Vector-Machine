from SVM import SVM
import matplotlib.pyplot as plt
import numpy as np


def draw(X, Y, color,marker='o'):
    plt.scatter(X, Y, color=color, marker=marker, alpha=0.5, linestyle='None', picker=True)


def drawline(X, Y, color,linestyle):
    plt.plot(X,Y,color=color,linestyle=linestyle)


def main():
    dim = 2
    N = 100
    np.random.seed(1000)
    data1 = np.random.multivariate_normal([-5, 10], [[2, 0], [0, 2]], int(N / 2))
    draw(data1[:, 0], data1[:, 1], "red")
    data2 = np.random.multivariate_normal([2, 2], [[2, 0], [0, 2]], int(N / 2))
    draw(data2[:, 0], data2[:, 1], "blue")
    # plt.show()
    X = np.empty(shape=[0, dim])
    X = np.append(X, data1, axis=0)
    X = np.append(X, data2, axis=0)
    svm = SVM(X)
    svm.train()
    print("train over")
    X,Y = svm.decision_boundary(0)
    drawline(X,Y,"black",'--')
    X,Y = svm.decision_boundary(+1)
    drawline(X,Y,"red",'-')
    X,Y = svm.decision_boundary(-1)
    drawline(X,Y,"blue",'-')
    # X,Y = svm.support_vector()
    # draw(X,Y,"black","x")

    plt.show()


if __name__ == "__main__":
    main()