import numpy as np


class SVM:
    def __init__(self, X, iter):
        self.b = 0
        self.dim = X.shape[1]
        self.w = np.zeros(shape=[self.dim])
        self.N = X.shape[0]
        self.X = X
        self.Y = [1] * int(self.N / 2) + [-1] * int(self.N / 2)
        self.Y = np.array(self.Y)
        self.alpha = np.random.random(self.N)
        self.hessian = np.empty(shape=[self.N, self.N])
        self.init_hessian()
        self.rate = 0.005
        self.C = 30
        self.iter = iter
        self.history_margin=np.array([])
        self.history_loss=np.array([])
        # print(self.X)
        # print(self.Y)
        print(self.alpha)

    def init_hessian(self):
        for i in range(0, self.N):
            for j in range(0, self.N):
                self.hessian[i][j] = self.Y[i] * self.Y[j] * np.dot(self.X[i], self.X[j])
    def pick(self):
        k = np.random.randint(0, self.N)
        while self.alpha[k] == 0:
            k = np.random.randint(0, self.N)
        return k

    def threshold(self, v):
        if v < 0:
            return 0
        if v > self.C:
            return self.C
        return v

    def train(self):
        f = np.ones(self.N)
        for iter in range(0, self.iter):
            d = f - self.hessian.dot(self.alpha)
            k = self.pick()
            dependent_sum = 0
            for i in range(0, self.N):
                if i != k:
                    self.alpha[i] = self.alpha[i] + self.rate * (d[i] + d[k] * (-self.Y[i] / self.Y[k]))
                    self.alpha[i] = self.threshold(self.alpha[i])
                    dependent_sum += self.alpha[i] * self.Y[i]
            self.alpha[k] = (-1 / self.Y[k]) * dependent_sum
            self.alpha[k] = self.threshold(self.alpha[k])
            self.getw()
            self.getb()
            # print(iter, self.margin())
            print(iter, self.loss())
            self.history_margin = np.append(self.history_margin,self.margin())
            self.history_loss = np.append(self.history_loss,self.loss())

        print("alpha", self.alpha)
        print("w", self.w)
        print("b", self.b)

        return self.alpha, self.w, self.b

    def getw(self):
        self.w = np.zeros(shape=[self.dim])
        for i in range(0, self.N):
            self.w += self.alpha[i] * self.Y[i] * self.X[i]

    def getb(self):
        count = 0
        self.b = 0
        for i in range(0, self.N):
            if self.alpha[i] > 0:
                self.b += (1 / self.Y[i]) - self.w.dot(self.X[i])
                count += 1
        self.b /= count

    def margin(self):
        return 2 / (self.w.dot(self.w))

    def loss(self):
        f = np.ones(self.N)
        return -0.5*(np.dot(np.dot(self.alpha.transpose(),self.hessian),self.alpha))+f.transpose().dot(self.alpha)

    def support_vector(self):
        X = []
        Y = []
        count = 0
        for i in range(0, self.N):
            if self.alpha[i] != 0:
                X.append(self.X[i][0])
                Y.append(self.X[i][1])
                print(self.X[i])
                count += 1
        print(count)
        return X, Y

    def decision_boundary(self, sign):
        X = []
        Y = []
        max = np.amax(self.X, axis=0)[0]
        min = np.amin(self.X, axis=0)[0]
        for x1 in np.arange(min, max, 0.01):
            x2 = ((-1) * (self.w[0] * x1 + self.b - sign)) / self.w[1]
            X.append(x1)
            Y.append(x2)
        return X, Y

