import numpy as np


class SVM:
    def __init__(self,X):
        self.b = 0
        self.dim = X.shape[1]
        self.w = np.zeros(shape=[self.dim])
        self.N = X.shape[0]
        self.X = X
        self.Y = [1]*int(self.N/2) + [-1]*int(self.N/2)
        self.Y = np.array(self.Y)
        self.alpha = np.random.random(self.N)
        self.rate = 0.002
        self.C = 50

    def pick(self):
        k = np.random.randint(0, self.N)
        while self.alpha[k] == 0:
            k = np.random.randint(0, self.N)
        return k

    def threshold(self,v):
        if v < 0:
            return 0
        if v > self.C:
            return self.C

    def train(self):
        f = np.ones(self.N)
        for iter in range(0, 100):
            hessian = np.empty(shape=[self.N, self.N])
            for i in range(0, self.N):
                for j in range(0, self.N):
                    hessian[i][j] = self.Y[i] * self.Y[j] * np.dot(self.X[i], self.X[j])
            d = f - hessian.dot(self.alpha)
            k = self.pick()
            dependent_sum = 0
            for i in range(0, self.N):
                if i != k:
                    self.alpha[i] = self.alpha[i] + self.rate * (d[i] - d[k] * self.Y[i] / self.Y[k])
                    dependent_sum += self.alpha[i] * self.Y[i]
                    self.alpha[i] = self.threshold(self.alpha[i])
            self.alpha[k] = (-1 / self.Y[k]) * dependent_sum
            self.alpha[k] = self.threshold(self.alpha[k])
            print(iter)

        self.getw()
        self.getb()
        return self.alpha,self.w,self.b


    def getw(self):
        for i in range(0, self.N):
            self.w += self.alpha[i] * self.Y[i] * self.X[i]

    def getb(self):
        count = 0
        for i in range(0, self.N):
            if self.alpha[i] > 0:
                # b+=1/Y[i]-np.multiply(w.transpose(),X[i])
                self.b += 1 / self.Y[i] - self.w.dot(self.X[i])
                count += 1
        self.b /= count


    def centerLine(self):
        X = []
        Y = []
        for x1 in np.arange(-2, 4, 0.01):
            x2 = (-self.w[0] * x1 - self.b) / self.w[1]
            X.append(x1)
            Y.append(x2)
        return X,Y

    def supportLine(self,sign):
        X = []
        Y = []
        for x1 in np.arange(-2, 4, 0.01):
            x2 = (-self.w[0] * x1 - self.b + sign) / self.w[1]
            X.append(x1)
            Y.append(x2)
        return