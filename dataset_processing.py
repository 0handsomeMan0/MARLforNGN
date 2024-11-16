import numpy as np
import pandas as pd
import copy
import torch
from data_set import DataSet
from user_info import UserInfo
import utils.dataset_utils as utils


def get_dataset(dataset_name='ml-1m'):
    """
    :param: args:
    :return: ratings: dataFrame ['user_id' 'movie_id' 'rating']
    :return: user_info:  dataFrame ['user_id' 'gender' 'age' 'occupation']
    """
    model_manager = utils.ModelManager('data_set')
    user_manager = utils.UserInfoManager(dataset_name)

    '''Do you want to clean workspace and retrain model/data_set user again?'''
    '''if you want to retrain model/data_set user, please set clean_workspace True'''
    model_manager.clean_workspace(True)
    user_manager.clean_workspace(True)

    # 导入模型信息
    try:
        ratings = model_manager.load_model(dataset_name + '-ratings')
        print("Load " + dataset_name + " data_set success.\n")
    except OSError:
        ratings = DataSet.LoadDataSet(name=dataset_name)
        model_manager.save_model(ratings, dataset_name + '-ratings')

    # 导入用户信息
    try:
        user_info = user_manager.load_user_info('user_info')
        print("Load " + dataset_name + " user_info success.\n")
    except OSError:
        user_info = UserInfo.load_user_info(name=dataset_name)
        user_manager.save_user_info(user_info, 'user_info')

    return ratings, user_info


def generate_zipf_distribution(size, a):
    """Generate indices based on Zipf-like distribution with parameter a"""
    ranks = np.arange(1, size + 1).astype(float)
    weights = ranks ** (-a)
    probabilities = weights / weights.sum()
    return np.random.choice(ranks, size=size, p=probabilities) - 1

def sampling(clients_num, gamma, dataset_name='ml-1m'):
    """
    :return: sample: matrix user_id|movie_id|rating|gender|age|occupation|label
    :return: user_group_train, the idx of sample for each client for training
    :return: user_group_test, the idx of sample for each client for testing
    """
    # 存储每个client信息
    model_manager = utils.ModelManager('clients')
    '''Do you want to clean workspace and retrain model/clients again?'''
    '''if you want to change test_size or retrain model/clients, please set clean_workspace True'''
    model_manager.clean_workspace(True)

    # 调用get_dataset函数，得到ratings,user_info
    ratings, user_info = get_dataset(dataset_name)
    users_num_client = int((user_info.index[-1] + 1) / clients_num)
    sample = pd.merge(ratings, user_info, on=['user_id'], how='inner')
    sample = sample.astype({'user_id': 'int64', 'movie_id': 'int64', 'rating': 'float64',
                            'gender': 'float64', 'age': 'float64', 'occupation': 'float64'})

    users_group_all, users_group_train, users_group_test = {}, {}, {}
    all_test_num = 0
    all_train_num = 0
    all_num = 0

    for i in range(clients_num):
        print('loading client ' + str(i))
        index_begin = ratings[ratings['user_id'] == int(users_num_client) * i + 1].index[0]
        index_end = ratings[ratings['user_id'] == users_num_client * (i + 1)].index[-1] if i != clients_num - 1 else \
        ratings.index[-1]
        users_group_all[i] = set(np.arange(index_begin, index_end + 1))

        # 生成Zipf分布的索引
        zipf_indices = generate_zipf_distribution(len(users_group_all[i]), gamma)
        zipf_indices = np.clip(zipf_indices, 1, len(users_group_all[i]))  # Clip to the maximum index
        zipf_indices = np.unique(zipf_indices)  # Ensure unique indices

        selected_indices = np.random.choice(list(users_group_all[i]), len(zipf_indices), replace=False)

        NUM_train = int(0.98 * len(selected_indices))
        users_group_train[i] = set(np.random.choice(selected_indices, NUM_train, replace=False))
        users_group_test[i] = set(selected_indices) - users_group_train[i]

        users_group_train[i] = list(users_group_train[i])
        users_group_test[i] = list(users_group_test[i])
        users_group_train[i].sort()
        users_group_test[i].sort()

        all_test_num += NUM_train / 0.98 * 0.2
        all_train_num += NUM_train
        all_num += int(len(users_group_all[i]))
        print('generate client ' + str(i) + ' info success\n')

    model_manager.save_model(sample, dataset_name + '-sample')
    model_manager.save_model(users_group_train, dataset_name + '-user_group_train')
    model_manager.save_model(users_group_test, dataset_name + '-user_group_test')

    return sample, users_group_train, users_group_test
