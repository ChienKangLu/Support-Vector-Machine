import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

def draw(X, Y, color):
    plt.scatter(X, Y, color=color, marker='o', alpha=0.5, linestyle='None', picker=True)

def drawline(X, Y, color,linestyle):
    plt.plot(X,Y,color=color,linestyle=linestyle)

dim = 2
N = 100

data1 = np.random.multivariate_normal([-3, 3], [[3, 0],[0, 5]],int(N/2))
draw(data1[:,0],data1[:,1],"red")
data2 = np.random.multivariate_normal([2, 2], [[0.2, 0],[0, 4]],int(N/2))
draw(data2[:,0],data2[:,1],"blue")
# plt.show()
X = np.empty(shape=[0,dim])
X = np.append(X, data1, axis=0)
X = np.append(X, data2, axis=0)

Y = [1]*int(N/2) + [-1]*int(N/2)
Y = np.array(Y)
print(Y)
# Y = np.random.randint(0,2,N)
# for y in np.nditer(Y,op_flags=['readwrite']):
#     if y == 0:
#         y[...]=-1
#
alpha = np.random.random(N)
print("alpha",alpha)
f = np.ones(N)
rate = 0.002
C = 50

for iter in range(0,100):
    hessian = np.empty(shape=[N,N])
    for i in range(0,N):
        for j in range(0,N):
            hessian[i][j] = Y[i]*Y[j]*np.dot(X[i],X[j])
    d = f - hessian.dot(alpha)
    k = np.random.randint(0, N)
    while alpha[k]==0:
        k = np.random.randint(0, N)
    depend_sum = 0
    for i in range(0,N):
        if i != k:
            alpha[i] = alpha[i]+rate*(d[i]-d[k]*Y[i]/Y[k])
            depend_sum += alpha[i]*Y[i]
            if alpha[i]<0:
                alpha[i] = 0
            if alpha[i]>C:
                alpha[i] = C
    alpha[k] = (-1/Y[k])*depend_sum
    if alpha[k]<0:
        alpha[k] = 0
    if alpha[k] > C:
        alpha[k] = C
    print(iter)
    # print("alpha",alpha)

w = np.zeros(shape=[dim])
for i in range(0,N):
    w += alpha[i]*Y[i]*X[i]
print(w)

b=0
count = 0
for i in range(0,N):
    if alpha[i]>0:
        # b+=1/Y[i]-np.multiply(w.transpose(),X[i])
        b+=1/Y[i]-w.dot(X[i])
        count+=1
b/=count
print(b)

XS = []
YS = []
for x1 in np.arange(-2, 4, 0.01):
    x2 = (-w[0]*x1-b)/w[1]
    XS.append(x1)
    YS.append(x2)
drawline(XS,YS,"black",'--')

XS = []
YS = []
for x1 in np.arange(-2, 4, 0.01):
    x2 = (-w[0]*x1-b+1)/w[1]
    XS.append(x1)
    YS.append(x2)
drawline(XS,YS,"red",'solid')

XS = []
YS = []
for x1 in np.arange(-2, 4, 0.01):
    x2 = ((-w[0]*x1-b)-1)/w[1]
    XS.append(x1)
    YS.append(x2)
drawline(XS,YS,"red",'solid')

q = np.empty(shape=[0,dim])
# print(q.shape)
for i in range(0,N):
    if alpha[i]>0:
        q = np.vstack([q,X[i]])
# print(q)
draw(q[:,0],q[:,1],"black")
plt.show()