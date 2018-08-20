from SVM import SVM
import matplotlib.pyplot as plt
import numpy as np


def draw(X, Y, color):
    plt.scatter(X, Y, color=color, marker='o', alpha=0.5, linestyle='None', picker=True)


def drawline(X, Y, color,linestyle):
    plt.plot(X,Y,color=color,linestyle=linestyle)


def main():
    dim = 2
    N = 100
    data1 = np.random.multivariate_normal([-3, 3], [[3, 0], [0, 5]], int(N / 2))
    draw(data1[:, 0], data1[:, 1], "red")
    data2 = np.random.multivariate_normal([2, 2], [[0.2, 0], [0, 4]], int(N / 2))
    draw(data2[:, 0], data2[:, 1], "blue")
    # plt.show()
    X = np.empty(shape=[0, dim])
    X = np.append(X, data1, axis=0)
    X = np.append(X, data2, axis=0)
    print(X)
    svm = SVM(X)
    svm.train()
    X,Y = svm.centerLine()
    drawline(X,Y,"black",'--')


if __name__ == "__main__":
    main()