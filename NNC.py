from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_utils import load_CIFAR10
import numpy as np 

import time
time1 = time.time()   #开始时间

class NearestNeighbor(object):
    def __init__(self):
        pass
    
    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype) #zeros表示创建一个0矩阵，dtype表示数据的类型

        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis = 1)   #L1距离
            #distances = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis = 1))   #L2距离
            #axis表示方向，=0代表列方向，=1代表行方向
            min_index = np.argmin(distances)   #argmin函数返回该方向上的最小值的下标
            Ypred[i] = self.ytr[min_index]

        return Ypred

Xtr, Ytr, Xte, Yte = load_CIFAR10('F:\\ComputerVision_LiFeiFei\\KNN_train\\cifar-10-batches-py')
#Xtr是一个四维的元组，50000x32x32x3
#Ytr是一个一维的元组，50000
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
#shape函数返回序列的维度
#shape[0]表示返回序列第一维的大小
#reshape就是改变序列的维度
#Xtr_rows是一个二维的元组，50000x3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)
# Xtr_rows = Xtr_rows[:5000,:]
# Xte_rows = Xte_rows[:1000,:]
# Ytr = Ytr[:5000]
# Yte = Yte[:1000]

nn = NearestNeighbor()
nn.train(Xtr_rows, Ytr)
Yte_predict = nn.predict(Xte_rows)

print('accuracy: %f' % (np.mean(Yte_predict == Yte)))

time2 = time.time()   #结束时间
print('time consuming: %f' % (time2 - time1))