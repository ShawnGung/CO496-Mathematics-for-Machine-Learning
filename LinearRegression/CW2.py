import numpy as np

import matplotlib.pyplot as plt

"""
sigma2
"""



def getTrainData(xRange,N):
    X = np.reshape(np.linspace(xRange[0],xRange[1],N),(N,1))
    y = np.cos(10*X**2) + 0.1*np.sin(100*X)
    return X,y

def plot(train_X,train_y,X,y,Y,xRange,type = 0):
    """
    :param X:
    :param y:
    :param Y:
    :param range:  X-axis range
    :return:
    """
    order = []
    label = []
    plt.suptitle('gaussian basis function')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    l1= plt.scatter(train_X, train_y, color='red',marker='x')  # 调用plot函数，这并不会立刻显示函数图像
    label.append(l1)
    order.append("data points")
    #plt.scatter(X, y, color='k', marker='x')  # 调用plot函数，这并不会立刻显示函数图像
    plt.xlim(xRange[0],xRange[1])
    plt.ylim(min(y) - 0.5 , max(y) + 0.5)
    color = ['b','g','y','c','k']


    Lambda = [0.0001*0.0001,0.5*0.5,4*4]
    for i in range(len(Y)):
        l1, = plt.plot(X, Y[i],color[i])  # 调用plot函数，这并不会立刻显示函数图像
        label.append(l1)
        if type == 0:
            # order.append("order:"+str(i))
            if i == 0:
                order.append("order:" + str(1))
            if i == 1:
                order.append("order:" + str(11))
        else:
            if i == 0:
                order.append("lambda:1e-8")
            else:
                order.append("lambda:"+str(round(Lambda[i],3)))
    plt.legend(label,order, loc='upper right')
    plt.show()


def MLE_theta(PHI,y):
    return np.linalg.pinv(PHI.T.dot(PHI)).dot(PHI.T).dot(y)


def MAP_theta(PHI,y,sigmas = [0.0001,0.5,4],b = 1):
    best_theta = []
    for sigma in sigmas:
        degreeOfI = PHI.shape[1]
        temp = np.linalg.pinv(PHI.T.dot(PHI) + ((sigma*sigma)/(b*b))*np.eye(degreeOfI)).dot(PHI.T).dot(y)
        best_theta.append(temp)
    return best_theta
    #return np.linalg.pinv(PHI.T.dot(PHI)).dot(PHI.T).dot(y)


def construct_PHI(order,X):
    """
    polynomial mapping
    :param order:
    :param X:
    :return:
    """
    PHI = []
    for j in range(order+1):
        temp = np.power(X,j)
        PHI.append(temp)
    return np.array(PHI).T

def tri_construct_PHI(order,X):
    """
    Trigonometric mapping
    :param order:
    :param X:
    :return:
    """
    PHI = []

    #bias
    PHI.append([1]*len(X))

    for j in range(1,order+1):
        temp1 = np.sin(2*np.pi*j*X)
        temp2 = np.cos(2*np.pi*j*X)
        PHI.append(temp1)
        PHI.append(temp2)
    return np.array(PHI).T


def gaussian_construct_PHI(order,X,mean = [],sigma = 0.1):
    """
    Trigonometric mapping
    :param order:
    :param X:
    :return:
    """
    PHI = []

    mean = np.reshape(np.linspace(0, 1, 20), (20, 1))

    #bias
    PHI.append([1]*len(X))

    for j in range(1,order+1):
        temp1 = np.exp(-1*(((X - mean[j-1])**2)/(2*sigma*sigma)))
        PHI.append(temp1)
    return np.array(PHI).T


def training(X,y,order = [0,1,2,3,11], func=construct_PHI):
    best_theta = []
    for o in order:
        PHI = func(o, X)  # (K+1) X (M)
        best_theta.append(MLE_theta(PHI, y))
    return best_theta

def fitting(best_theta,X,order = [0,1,2,3,11],func = construct_PHI):
    """
    :param best_theta:
    :param X:
    :param order:
    :param func: different mapping function
    :return:
    """
    Y = []
    for i in range(len(order)):
        PHI = func(order[i], X)
        Y.append(PHI.dot(best_theta[i]))
    return np.array(Y)


def normal_order(train_X,train_y):
    """
    a)
    :param train_X:
    :param train_y:
    :param test_X:
    :return:
    """

    # test_data
    test_X, test_y = getTrainData([-0.3,1.3], 200)
    test_X = np.array(test_X).reshape(-1)
    test_y = np.array(test_y).reshape(-1)

    best_theta = training(train_X, train_y,order = [0,1,2,3,11],func = construct_PHI)
    Y = fitting(best_theta, test_X,order = [0,1,2,3,11],func = construct_PHI)
    plot(train_X,train_y,test_X, test_y, Y,[-0.3-0.3,1.3+0.3])


def leave_one_out(train_X,train_y,k):
    valid_X = train_X[k]
    training_X = np.append(train_X[:k],train_X[k+1:])  # training_X is the part of train_X for training
    valid_y = train_y[k]
    training_Y = np.append(train_y[:k],train_y[k+1:])
    return training_X,training_Y,np.array([valid_X]),np.array([valid_y])



def trigonometric_order(train_X,train_y):
    """
    a)
    :param train_X:
    :param train_y:
    :param test_X:
    :return:
    """
    # test_data
    test_X, test_y = getTrainData([-1, 1.2], 200)
    test_X = np.array(test_X).reshape(-1)
    test_y = np.array(test_y).reshape(-1)

    # b)
    best_theta = training(train_X, train_y,order = [1,11],func = tri_construct_PHI)
    Y = fitting(best_theta, test_X,order = [1,11] ,func = tri_construct_PHI)


    plot(train_X,train_y,test_X, test_y, Y,[-1-0.3, 1.2+0.3])


    # # c)
    # k_errors = []
    # order = [i for i in range(11)]
    # #sigma^2
    # all_sigma2 = []
    # #leave-one-out cross validation
    # for k in range(len(train_X)):
    #     training_X, training_y, valid_X, valid_y = leave_one_out(train_X,train_y,k)
    #     best_theta = training(training_X, training_y, order=order, func=tri_construct_PHI)
    #     Y = fitting(best_theta, valid_X, order=order, func=tri_construct_PHI)
    #     #errors for all orders
    #     errors = (Y - valid_y)**2
    #     k_errors.append(errors.reshape(-1))
    #     # sigma^2
    #
    #
    #     #for each training_X, counting for all orders
    #     each_order_sigma2 = []
    #     for i in range(len(order)):
    #         func = tri_construct_PHI
    #         PHI = func(order[i], training_X)  # (K+1) X (M)
    #         term = training_y - PHI.dot(best_theta[i])
    #         sigma2 = term ** 2
    #         each_order_sigma2.append(np.sum(sigma2) / len(training_X)) # 1 / N (N = 24)
    #     all_sigma2.append(each_order_sigma2)
    # all_sigma2 = np.array(all_sigma2)
    # all_sigma2 = np.average(all_sigma2,axis=0) # 1 / 25
    #
    #
    #
    # k_errors = np.array(k_errors)
    # k_errors_average = np.average(k_errors,axis=0)
    # plt.xlabel('order')
    # plt.ylabel('average_errors')
    # # 设置坐标轴刻度
    # my_x_ticks = np.arange(0, 10, 1)
    # plt.xticks(my_x_ticks)
    # label = []
    # labelname = []
    # l1, = plt.plot(order, k_errors_average)
    # label.append(l1)
    # labelname.append("test errors")
    # l2, = plt.plot(order, all_sigma2)
    # label.append(l2)
    # labelname.append("MLE of sgima^2")
    # plt.legend(label, labelname, loc='upper right')
    # plt.show()

def gaussian_order(train_X,train_y):
    """
    a)
    :param train_X:
    :param train_y:
    :param test_X:
    :return:
    """

    # test_data
    test_X, test_y = getTrainData([-0.3,1.3], 200)
    test_X = np.array(test_X).reshape(-1)
    test_y = np.array(test_y).reshape(-1)

    Y = []
    #train_case
    PHI = gaussian_construct_PHI(20, train_X)  # (K+1) X (M)
    best_theta = MAP_theta(PHI, train_y)

    for i in range(3):
        # test_case
        PHI = gaussian_construct_PHI(20, test_X)  # (K+1) X (M)
        theta = np.array(best_theta[i])
        Y.append(PHI.dot(theta))
    plot(train_X,train_y,test_X, test_y, Y,[-0.3-0.3,1.3+0.3],1)



# train_Data
train_X,train_y = getTrainData([0,0.9],25)
train_X = np.array(train_X).reshape(-1)
train_y = np.array(train_y).reshape(-1)
#normal_order(train_X,train_y)
gaussian_order(train_X,train_y)