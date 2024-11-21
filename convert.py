import utils
import numpy as np
import pandas as pd
import scipy.sparse as sp

def convert_csv_to_npy(csv_path, save_path):
    # 使用utils.py中的loadsparse加载CSV
    #features = utils.loadsparse(csv_path)
    # 或者使用loaddata加载非稀疏数据
    features = utils.loaddata(csv_path)

    # 转换为数组
    if isinstance(features, sp.csr_matrix):
        features = features.toarray()

    # 标准化
    features = utils.normalize(features)

    # 分割为训练集和测试集
    train_size = int(len(features) * 0.7)  # 70%训练
    X_train = features[:train_size]
    X_test = features[train_size:]

    # 调整维度 [N, L, M] -> [N, M, L]
    X_train = np.transpose(X_train, axes=(0, 2, 1))
    X_test = np.transpose(X_test, axes=(0, 2, 1))

    # 保存
    np.save(f'{save_path}/X_train.npy', X_train)
    np.save(f'{save_path}/X_test.npy', X_test)

    return X_train, X_test

convert_csv_to_npy('data/PenDigits/dataset_32_pendigits.csv', 'data/PenDigits/')