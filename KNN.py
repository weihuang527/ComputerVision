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

    def predict(self, X, k=1):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype) #zeros表示创建一个0矩阵，dtype表示数据的类型

        for i in range(num_test):
            #distances = np.sum(np.abs(self.Xtr - X[i, :]), axis = 1)   #L1距离
            distances = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis = 1))   #L2距离
            #axis表示方向，=0代表列方向，=1代表行方向
            min_index = np.argsort(distances)   #argsort函数按从小到大的顺序返回索引号
            classCount = {}
            for j in range(k):
                voteIlabel = self.ytr[min_index[j]]
                classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
            sortedClassCount = sorted(classCount.items(), key = lambda c:c[1], reverse = True)
            #sorted函数对所有可迭代的对象进行排序操作，与sort的区别在于sort在于原对象上排序，而sorted是排序生成新对象，
            #不改变原对象
            #items函数返回一个元组数组，在python2.x中是函数iteritems
            #key = c:c[1]以字典的值为排序对象
            #默认为升序，reverse = True表示降序
            Ypred[i] = sortedClassCount[0][0]

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

Xval_rows = Xtr_rows[:100,:]   #前100张图片用于验证集
Yval = Ytr[:100]
Xtr_rows = Xtr_rows[100:5000,:]  #100到5000用于训练集
Xte_rows = Xte_rows[100:1000,:]
Ytr = Ytr[:5000]
Yte = Yte[:1000]

validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:
    nn = NearestNeighbor()
    nn.train(Xtr_rows, Ytr)
    Yval_predict = nn.predict(Xval_rows, k=k)
    acc = np.mean(Yval_predict == Yval)
    print('k= %d, accuracy: %f' % (k, acc,))

    validation_accuracies.append((k, acc))

time2 = time.time()   #结束时间
print('time consuming: %f' % (time2 - time1))