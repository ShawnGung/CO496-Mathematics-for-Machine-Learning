# -*- coding: utf-8 -*-

"""
Use this file for your answers.

This file should been in the root of the repository
(do not move it or change the file name)

"""

# NB this is tested on python 2.7. Watch out for integer division

import numpy as np

import matplotlib.pyplot as plt
from scipy import optimize


def grad_f1(x):
    """
    4 marks

    :param x: input array with shape (2, )
    :return: the gradient of f1, with shape (2, )
    """
    return np.array([8 * x[0] - 2 * x[1] - 1, 8 * x[1] - 2 * x[0] - 1])


def grad_f2(x):
    """
    6 marks

    :param x: input array with shape (2, )
    :return: the gradient of f2, with shape (2, )
    """
    # inner_term = (x[0]-1)**2 + x[1]**2
    # cos_term = np.cos(inner_term)
    # gradient = [6*x[0]-2*x[1]-2+2*(x[0] - 1)*cos_term , 6*x[1]-2*x[0]+6+cos_term*2*x[1]]

    a = np.array([1, 0])
    B = np.array([[3, -1], [-1, 3]])
    b = np.array([0, -1])
    gradient = (np.cos((x - a).T.dot(x - a))) * 2 * (x.T - a.T) + 2 * (x.T - b.T).dot(B)

    return np.array(gradient)


def f2(x):
    a = np.array([1, 0])
    B = np.array([[3, -1], [-1, 3]])
    b = np.array([0, -1])

    f2 = np.sin((x - a).T.dot(x - a)) + (x - b).T.dot(B).dot(x - b)
    return f2


def grad_f3(x):
    """
    This question is optional. The test will still run (so you can see if you are correct by
    looking at the testResults.txt file), but the marks are for grad_f1 and grad_f2 only.

    Do not delete this function.

    :param x: input array with shape (2, )
    :return: the gradient of f3, with shape (2, )
    """
    a = np.array([1, 0])
    B = np.array([[3, -1], [-1, 3]])
    b = np.array([0, -1])
    I = np.array([[1, 0], [0, 1]])
    first_inner_term = -1 * (x - a).T.dot(x - a)
    second_inner_term = -1 * (x - b).T.dot(B).dot(x - b)
    det_derivative = np.array([1.0 * x[0] / 50, 1.0 * x[1] / 50])
    x_re = np.reshape(x, [2, 1])
    third_log_term = np.linalg.det((1.0 / 100) * I + x_re.dot(x_re.T))
    gradient = -1 * np.exp(first_inner_term) * (-2) * (x.T - a.T) + np.exp(second_inner_term) * (2) * (x - b).T.dot(
        B) + (1.0 / 10) * det_derivative / third_log_term

    return np.array(gradient)


def f3(x):
    a = np.array([1, 0])
    B = np.array([[3, -1], [-1, 3]])
    b = np.array([0, -1])
    I = np.array([[1, 0], [0, 1]])
    x_re = np.reshape(x, [2, 1])
    f = np.exp(-(x - a).T.dot(x - a))
    s = np.exp(-(x - b).T.dot(B).dot(x - b))
    t = (-1.0/10)*np.log(np.linalg.det((1.0/100)*I + x_re.dot(x_re.T)))
    return 1-(f+s+t)

def plot_grad_f2():
    orix = np.array([1, -1])
    max_iteration = 50
    #step_size = 0.1

    plt.figure(figsize=(12,8))
    plt.suptitle('f2 gradient descend in terms of different step sizes')
    for k in range(1, 11):
        x = orix
        step_size = k * 0.1

        resultx = [x[0]]
        resulty = [x[1]]
        resultVy = [f2(x)]
        resultVx = [0]
        for i in range(max_iteration):
            x = x - step_size * grad_f2(x)
            resultx.append(x[0])
            resulty.append(x[1])
            resultVy.append(f2(x))
            resultVx.append(i+1)

        xlinspace = [-2,2]
        ylinspace = [-2,0]
        newX = []
        newY = []
        if max(max(resultx),-min(resultx)) > 100 and max(max(resulty),-min(resulty)) > 100:
            xlinspace = [-25,25]
            ylinspace = [-25,25]

        for i in range(len(resultx)):
            if abs(resultx[i]) <100 and abs(resulty[i]) <100:
                newX.append(resultx[i])
                newY.append(resulty[i])
        resultx = newX
        resulty = newY




        X = np.linspace(xlinspace[0], xlinspace[1], 50)
        Y = np.linspace(ylinspace[0], ylinspace[1], 50)
        X,Y = np.meshgrid(X,Y)

        a = np.array([1, 0])
        B = np.array([[3, -1], [-1, 3]])
        b = np.array([0, -1])


        Z = np.zeros([50,50])
        for i in range(50):
            for j in range(50):
                x = np.array([X[i,j],Y[i,j]])
                Z[i,j] = np.sin((x - a).T.dot(x - a)) + (x - b).T.dot(B).dot(x - b)

        plt.subplot(4,3,k)
        plt.tight_layout(rect = [0, 0.03, 1, 0.95])
        plt.title('step size:'+str(round(step_size,2)))  # 标题
        ct = plt.contour(X,Y,Z,20)  # 调用plot函数，这并不会立刻显示函数图像

        plt.plot(resultx,resulty,color='red')

        plt.xlim(xlinspace[0],xlinspace[1])
        plt.ylim(ylinspace[0],ylinspace[1])
        plt.clabel(ct,inline = True, fontsize = 7,colors = 'blue')
        plt.xlabel('x')  # 使用xlable、ylable函数添加标签
        plt.ylabel('y')
    plt.show()

    # plt.subplot(122)
    # plt.title('f2 gradient descend with step size:0.12')  # 标题
    # plt.plot(resultVx, resultVy)  # 调用plot函数，这并不会立刻显示函数图像
    # plt.xlabel('times of iteration')  # 使用xlable、ylable函数添加标签
    # plt.ylabel('f2(x)')
    # plt.savefig('f2_' + str(step_size * 100) + '.PNG')  # 调用show函数显示函数图像



def plot_grad_f3():
    orix = np.array([1, -1])
    max_iteration = 50
    #step_size = 0.1

    plt.figure(figsize=(15, 8))
    plt.suptitle('f3 gradient descend in terms of different step sizes')
    for k in range(1, 11):
        x = orix
        step_size = k * 0.1
        # plot_grad_f2(step_size)
        # plot_grad_f3(step_size)

        resultx = [x[0]]
        resulty = [x[1]]
        resultVy = [f2(x)]
        resultVx = [0]
        for i in range(max_iteration):
            x = x - step_size * grad_f3(x)
            resultx.append(x[0])
            resulty.append(x[1])
            resultVy.append(f3(x))
            resultVx.append(i+1)

        X = np.linspace(-2, 2, 100)
        Y = np.linspace(-2, 2, 100)
        X,Y = np.meshgrid(X,Y)


        Z = np.zeros([100, 100])
        for i in range(100):
            for j in range(100):
                x = np.array([X[i, j], Y[i, j]])
                Z[i, j] = f3(x)



        plt.subplot(4,3,k)
        plt.tight_layout(rect = [0, 0.03, 1, 0.95])
        #plt.title('f3 gradient descend with step size:0.'+str(step_size*100))  # 标题
        plt.title('step size:' + str(round(step_size, 2)))  # 标题
        ct =plt.contour(X,Y,Z,20)  # 调用plot函数，这并不会立刻显示函数图像
        plt.clabel(ct,inline = True, fontsize = 7,colors = 'blue')
        plt.plot(resultx, resulty,color='red')  # 调用plot函数，这并不会立刻显示函数图像
        plt.xlabel('x')  # 使用xlable、ylable函数添加标签
        plt.ylabel('y')

    plt.show()


def plot_grad_f2_fixed_stepsize():
    orix = np.array([1, -1])
    max_iteration = 50
    #step_size = 0.1

    plt.figure(figsize=(12,8))
    plt.suptitle('f2 gradient descend with the stepsize : 0.12')
    x = orix
    step_size = 0.11

    resultx = [x[0]]
    resulty = [x[1]]
    resultVy = [f2(x)]
    resultVx = [0]
    for i in range(max_iteration):

        print(f2(x))
        x = x - step_size * grad_f2(x)
        resultx.append(x[0])
        resulty.append(x[1])
        resultVy.append(f2(x))
        resultVx.append(i+1)

    xlinspace = [-2,2]
    ylinspace = [-2,0]
    newX = []
    newY = []
    if max(max(resultx),-min(resultx)) > 100 and max(max(resulty),-min(resulty)) > 100:
        xlinspace = [-25,25]
        ylinspace = [-25,25]

    for i in range(len(resultx)):
        if abs(resultx[i]) <100 and abs(resulty[i]) <100:
            newX.append(resultx[i])
            newY.append(resulty[i])
    resultx = newX
    resulty = newY




    X = np.linspace(xlinspace[0], xlinspace[1], 50)
    Y = np.linspace(ylinspace[0], ylinspace[1], 50)
    X,Y = np.meshgrid(X,Y)

    a = np.array([1, 0])
    B = np.array([[3, -1], [-1, 3]])
    b = np.array([0, -1])


    Z = np.zeros([50,50])
    for i in range(50):
        for j in range(50):
            x = np.array([X[i,j],Y[i,j]])
            Z[i,j] = np.sin((x - a).T.dot(x - a)) + (x - b).T.dot(B).dot(x - b)

    plt.tight_layout(rect = [0, 0.03, 1, 0.95])
    plt.title('step size:'+str(round(step_size,2)))  # 标题
    ct = plt.contour(X,Y,Z,40)  # 调用plot函数，这并不会立刻显示函数图像

    plt.plot(resultx,resulty,color='red')

    plt.xlim(xlinspace[0],xlinspace[1])
    plt.ylim(ylinspace[0],ylinspace[1])
    plt.clabel(ct,inline = True, fontsize = 7,colors = 'blue')
    plt.xlabel('x')  # 使用xlable、ylable函数添加标签
    plt.ylabel('y')
    plt.show()

    # plt.subplot(122)
    # plt.title('f2 gradient descend with step size:0.12')  # 标题
    # plt.plot(resultVx, resultVy)  # 调用plot函数，这并不会立刻显示函数图像
    # plt.xlabel('times of iteration')  # 使用xlable、ylable函数添加标签
    # plt.ylabel('f2(x)')
    # plt.savefig('f2_' + str(step_siz


def plot_grad_f3_fixed_stepsize():
    orix = np.array([0,0])
    max_iteration = 50
    #step_size = 0.1

    plt.figure(figsize=(15, 8))
    plt.suptitle('f3 gradient descend with the stepsize:0.12')
    x = orix
    step_size = 0.1
    # plot_grad_f2(step_size)
    # plot_grad_f3(step_size)

    resultx = [x[0]]
    resulty = [x[1]]
    resultVy = [f2(x)]
    resultVx = [0]
    for i in range(max_iteration):
        print(f3(x))
        x = x - step_size * grad_f3(x)
        resultx.append(x[0])
        resulty.append(x[1])
        resultVy.append(f3(x))
        resultVx.append(i+1)

    X = np.linspace(-2, 2, 100)
    Y = np.linspace(-2, 2, 100)
    X,Y = np.meshgrid(X,Y)


    Z = np.zeros([100, 100])
    for i in range(100):
        for j in range(100):
            x = np.array([X[i, j], Y[i, j]])
            Z[i, j] = f3(x)



    plt.tight_layout(rect = [0, 0.03, 1, 0.95])
    #plt.title('f3 gradient descend with step size:0.'+str(step_size*100))  # 标题
    plt.title('step size:' + str(round(step_size, 2)))  # 标题
    ct =plt.contour(X,Y,Z,40)  # 调用plot函数，这并不会立刻显示函数图像
    plt.clabel(ct,inline = True, fontsize = 7,colors = 'blue')
    plt.plot(resultx, resulty,color='red')  # 调用plot函数，这并不会立刻显示函数图像
    plt.xlabel('x')  # 使用xlable、ylable函数添加标签
    plt.ylabel('y')

    plt.show()

# 通过scipy 来求最小值
# allMin = []

plot_grad_f3_fixed_stepsize()
# optimize.fmin_slsqp(f2, np.array([1, -1]))


#plot_grad_f3(0.12)