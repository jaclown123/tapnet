import utils
import numpy as np
import pandas as pd
import scipy.sparse as sp

def convert_csv_to_npy(csv_path, save_path):
    # 使用utils.py中的loadsparse加载CSV
    #features = utils.loadsparse(csv_path)
    # 或者使用loaddata加载非稀疏数据
    features = utils.loaddata(csv_path)
    X = features[1:, :-1]
    y = features[1:, -1]
    #y = utils.encode_onehot(y)
    print(X.shape, y.shape)
    # 转换为数组
    X = np.array(X)
    # 标准化
    #X = utils.normalize(X)

    # 分割为训练集和测试集
    train_size = int(len(features) * 0.7)  # 70%训练
    print(train_size)
    X_train = X[:train_size]
   # X_train = X_train.reshape(5, int(7695/5), 17)
    X_test = X[train_size:]
    y_train = y[:train_size]
   # y_train = y_train.reshape(5, int(7695/5), 10)
    y_test = y[train_size:]
    # print("Before reshape:")
    # print(X[:5])  # 查看原始数据
    # print(y[:5])
    #
    # print("After reshape:")
    # print(X_train[0, :5])  # 查看reshape后的数据
    # print(y_train[0, :5])


    # 保存
    np.save(f'{save_path}/X_train.npy', X_train)
    np.save(f'{save_path}/X_test.npy', X_test)
    np.save(f'{save_path}/y_train.npy', y_train)
    np.save(f'{save_path}/y_test.npy', y_test)
    return X_train, X_test, y_train, y_test

convert_csv_to_npy('data/PenDigits/dataset_32_pendigits.csv', 'data/PenDigits/')