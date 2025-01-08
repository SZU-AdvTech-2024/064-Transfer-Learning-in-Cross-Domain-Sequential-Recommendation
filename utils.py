import sys
import copy
import os
from datetime import datetime

import pandas as pd
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import pickle
from datetime import datetime
import pytz




def random_neq(all_items, interacted_items):
    t = np.random.choice(all_items, 1, replace=False)[0]
    while t in interacted_items:
        t = np.random.choice(all_items, 1, replace=False)[0]
    return t

# def random_neq(l,r,s):
#     t = np.random.randint(l, r)
#     while t in s:
#         t = np.random.randint(l, r)
#     return t

def sample_function(user_train, user_train2, user_train3, time1, time2, time3, usernum, itemnum,target_domain_all_items, batch_size, maxlen,
                    result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        seq2 = np.zeros([maxlen], dtype=np.int32)
        seq3 = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        t1 = np.zeros([maxlen], dtype=np.int32)
        t2 = np.zeros([maxlen], dtype=np.int32)
        t3 = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i, t in reversed(list(zip(user_train[user][:-1], time1[user][:-1]))):
            seq[idx] = i
            t1[idx] = t
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(target_domain_all_items, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        idx = maxlen - 1
        for i, t in reversed(list(zip(user_train2[user][:-1], time2[user][:-1]))):
            seq2[idx] = i
            t2[idx] = t
            idx -= 1
            if idx == -1: break

        idx = maxlen - 1
        for i, t in reversed(list(zip(user_train3[user][:-1], time3[user][:-1]))):
            seq3[idx] = i
            t3[idx] = t
            idx -= 1
            if idx == -1: break

        mask2 = np.zeros([maxlen], dtype=np.int32)
        idx2 = 0
        for idx in range(len(seq)):
            while idx2 < maxlen and t1[idx] >= t2[idx2]:
                idx2 += 1
            mask2[idx] = idx2

        mask3 = np.zeros([maxlen], dtype=np.int32)
        idx3 = 0
        for idx in range(len(seq)):
            while idx3 < maxlen and t1[idx] >= t3[idx3]:
                idx3 += 1
            mask3[idx] = idx3

        return (user, seq, pos, neg, seq2, mask2, seq3, mask3)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, User2, User3, time1, time2, time3, usernum, itemnum,target_domain_all_items, batch_size=64, maxlen=10,
                 n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):  
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      User2,
                                                      User3,
                                                      time1,
                                                      time2,
                                                      time3,
                                                      usernum,
                                                      itemnum,
                                                      target_domain_all_items,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def get_sequences(Users,domain_data,obj):
    sequences={}
    for (domain,user_id), group in domain_data:
        if domain not in sequences:
            sequences[domain] = {}
        sequences[domain][user_id] = group[obj].tolist()
    # 有些用户没有训练集（设置为空）
    for domain in sequences:
        non_user=list(set(Users)-set(sequences[domain].keys()))
        for user_id in non_user:
            sequences[domain][user_id] = []
    return sequences

def convert_to_timestamp(date_time_str):
    """
    将指定格式的日期时间字符串转换为时间戳

    :param date_time_str: 日期时间字符串，格式如 "2019-11-22 16:05:41+00:00"
    :return: 对应的时间戳（以秒为单位）
    """
    # 将日期时间字符串解析为datetime对象，并设置时区
    dt = datetime.fromisoformat(date_time_str)
    # 计算时间戳
    timestamp = dt.timestamp()

    return timestamp




def data_partition_time(target_domain,source_domain1,source_domain2,is_only_overlap,is_full_candidate,dataset_file):
    # 读取数据
    dtype_dict = {'user_id': str, 'item_id': str, 'timestamp':str, 'category_id':str,'categories':str, 'domain': str}
    total_data_train=pd.read_csv("cross_data/"+dataset_file+"/train_data.csv",header=0,names=['user_id','item_id','timestamp','category_id','categories','domain'],dtype=dtype_dict)
    total_data_valid=pd.read_csv("cross_data/"+dataset_file+"/valid_data.csv",header=0,names=['user_id','item_id','timestamp','category_id','categories','domain'],dtype=dtype_dict)
    total_data_test=pd.read_csv("cross_data/"+dataset_file+"/test_data.csv",header=0,names=['user_id','item_id','timestamp','category_id','categories','domain'],dtype=dtype_dict)

    # # 把实际时间改为时间戳
    if dataset_file=='Rees46':
        total_data_train['timestamp']=total_data_train['timestamp'].apply(convert_to_timestamp)
        total_data_valid['timestamp']=total_data_valid['timestamp'].apply(convert_to_timestamp)
        total_data_test['timestamp']=total_data_test['timestamp'].apply(convert_to_timestamp)


    total_data=pd.concat([total_data_train, total_data_valid, total_data_test])


    file_path = "cross_data/"+dataset_file+"/dataset_"+target_domain+"_"+ str(is_only_overlap)+"_"+str(is_full_candidate) + "_.pickle"
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
            print("read",file_path)
            return dataset

    if (is_only_overlap):
        ### 只考虑重叠用户

        # 找到重叠用户
        domain_count_by_user=total_data.groupby('user_id')['domain'].nunique()
        overlap_user=domain_count_by_user[domain_count_by_user==3].index
        overlap_user_num=len(overlap_user)
        print("Overlapping users in three domain:", overlap_user_num)

        #筛选出重叠用户的记录
        total_data_train=total_data_train[total_data_train['user_id'].isin(overlap_user)]
        total_data_valid=total_data_valid[total_data_valid['user_id'].isin(overlap_user)]
        total_data_test=total_data_test[total_data_test['user_id'].isin(overlap_user)]
        total_data=total_data[total_data['user_id'].isin(overlap_user)]

    # 打印每个域的用户，物品数量
    if(not is_only_overlap):
        user_count_by_domain=total_data.groupby('domain')['user_id'].unique()
        for domain, users in user_count_by_domain.items():
            print(f"Domain: {domain}, Item num: {len(set(users))}")
        user_num1=user_count_by_domain[target_domain]
        user_num2=user_count_by_domain[source_domain1]
        user_num3=user_count_by_domain[source_domain2]
    item_by_domain=total_data.groupby('domain')['item_id'].unique()
    for domain, items in item_by_domain.items():
        print(f"Domain: {domain}, Item num: {len(set(items))}")
    item_num1=len(item_by_domain[target_domain])
    item_num2=len(item_by_domain[source_domain1])
    item_num3=len(item_by_domain[source_domain2])


    # 重新映射userID和itemID
    Users = total_data['user_id'].unique()
    Items = total_data['item_id'].unique()
    print("user_num:", len(Users),"items:", len(Items))
    total_user_num=len(Users)

    user_map={old_id:new_id for new_id, old_id in enumerate(set(Users),1)}
    item_map={old_id:new_id for new_id, old_id in enumerate(set(Items),1)}


    total_data['user_id']=total_data['user_id'].map(user_map)
    total_data_train['user_id']=total_data_train['user_id'].map(user_map)
    total_data_valid['user_id']=total_data_valid['user_id'].map(user_map)
    total_data_test['user_id']=total_data_test['user_id'].map(user_map)

    total_data['item_id']=total_data['item_id'].map(item_map)
    total_data_train['item_id']=total_data_train['item_id'].map(item_map)
    total_data_valid['item_id']=total_data_valid['item_id'].map(item_map)
    total_data_test['item_id']=total_data_test['item_id'].map(item_map)

    Users = total_data['user_id'].unique()
    Items = total_data['item_id'].unique()

    # 根据不同的域，分离每个用户每个域的物品序列，时间戳序列
    domain_data=total_data.groupby(['domain','user_id'])
    domain_data_train=total_data_train.groupby(['domain','user_id'])
    domain_data_valid=total_data_valid.groupby(['domain','user_id'])
    domain_data_test=total_data_test.groupby(['domain','user_id'])

    item_sequences_train=get_sequences(Users,domain_data_train,'item_id')
    item_sequences_valid=get_sequences(Users,domain_data_valid,'item_id')
    item_sequences_test=get_sequences(Users,domain_data_test,'item_id')
    time_sequences=get_sequences(Users,domain_data,'timestamp')



    #生成负样本候选集
    user_neg1={}
    # 目标域每个用户交互过的物品
    target_domain_data = total_data[total_data['domain'] == target_domain]
    target_domain_interacted_items = target_domain_data.groupby('user_id')['item_id'].unique()
    target_domain_all_items = target_domain_data['item_id'].unique()
    target_domain_all_items_set = set(target_domain_all_items)
    if is_full_candidate:
        # file_path = "cross_data/"+dataset_file+"/"+target_domain+"_full_neg_data.pickle"
        # if os.path.exists(file_path):
        #     with open(file_path, 'rb') as f:
        #         target_domain_uninteracted_items = pickle.load(f)
        #         print("read",file_path)
        # else:
        #     target_domain_uninteracted_items = {}
        #     for  user_id, interacted_items in target_domain_interacted_items.items():
        #         target_domain_interacted_items_set = set(interacted_items)
        #         target_domain_uninteracted_items[user_id] = list(target_domain_all_items_set - target_domain_interacted_items_set)
        #
        #     #保存变量
        #     file_path = "cross_data/"+dataset_file+"/"+target_domain+"_full_neg_data.pickle"
        #     with open(file_path, 'wb') as f:
        #         pickle.dump(target_domain_uninteracted_items, f)
        #         print("save",file_path)
        for user_id in Users:
            user_neg1[user_id]=list(target_domain_all_items_set)

    else:
        file_path = "cross_data/" +dataset_file+"/"+ target_domain + "_random_100_neg_data.pickle"
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                target_domain_random_100_uninteracted_items=pickle.load(f)
                print("read", file_path)
        else:
            target_domain_random_100_uninteracted_items={}
            target_users=target_domain_data['user_id'].unique()
            #随机选100个
            for user_id in target_users:
                interacted_items_set=set(target_domain_interacted_items[user_id])
                target_domain_random_100_uninteracted_items[user_id]=(
                    np.random.choice(
                    list(target_domain_all_items_set-interacted_items_set) ,
                    size=100,
                    replace=False))

            # 保存变量
            file_path = "cross_data/" +dataset_file+"/"+ target_domain + "_random_100_neg_data.pickle"
            with open(file_path, 'wb') as f:
                pickle.dump(target_domain_random_100_uninteracted_items, f)
                print("save", file_path)

    # 按原来的格式分离三个域的数据
    usernum = overlap_user_num if is_only_overlap else total_user_num

    user_train1=item_sequences_train[target_domain]
    user_valid1=item_sequences_valid[target_domain]
    user_test1=item_sequences_test[target_domain]
    time1=time_sequences[target_domain]

    user_train2=item_sequences_train[source_domain1]
    user_valid2=item_sequences_valid[source_domain1]
    user_test2=item_sequences_test[source_domain1]
    time2=time_sequences[source_domain1]

    user_train3=item_sequences_train[source_domain2]
    user_valid3=item_sequences_valid[source_domain2]
    user_test3=item_sequences_test[source_domain2]
    time3=time_sequences[source_domain2]



    if not is_full_candidate:
        user_neg1=target_domain_random_100_uninteracted_items

    dataset=[user_train1, user_valid1, user_test1, usernum, item_num1, user_neg1, user_train2, user_valid2, user_test2,
            item_num2, user_train3, user_valid3, user_test3, item_num3, time1, time2, time3, target_domain_all_items]

    file_path = "cross_data/"+dataset_file+"/dataset_"+target_domain+"_"+ str(is_only_overlap)+"_"+str(is_full_candidate) + "_.pickle"
    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)
        print("save", file_path)


    return dataset


def data_partition(fname, fname2, fname3):
    usernum = 0
    itemnum1 = 0
    User = defaultdict(list)
    User1 = defaultdict(list)
    User2 = defaultdict(list)
    User3 = defaultdict(list)
    user_train1 = {}
    user_valid1 = {}
    user_test1 = {}
    neglist1 = defaultdict(list)
    user_neg1 = {}

    itemnum2 = 0
    user_train2 = {}
    user_valid2 = {}
    user_test2 = {}

    itemnum3 = 0
    user_train3 = {}
    user_valid3 = {}
    user_test3 = {}

    user_map = dict()
    item_map = dict()

    user_ids = list()
    item_ids1 = list()
    item_ids2 = list()
    item_ids3 = list()

    Time = defaultdict(list)
    Time1 = {}
    Time2 = {}
    Time3 = {}

    f = open('cross_data/processed_data_all/%s_train.csv' % fname, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        user_ids.append(u)
        item_ids1.append(i)
        User[u].append(i)
        Time[u].append(t)

    f = open('cross_data/processed_data_all/%s_valid.csv' % fname, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        user_ids.append(u)
        item_ids1.append(i)
        User[u].append(i)
        Time[u].append(t)

    f = open('cross_data/processed_data_all/%s_test.csv' % fname, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        user_ids.append(u)
        item_ids1.append(i)
        User[u].append(i)
        Time[u].append(t)

    for u in user_ids:
        if u not in user_map:
            user_map[u] = usernum + 1
            usernum += 1
    for i in item_ids1:
        if i not in item_map:
            item_map[i] = itemnum1 + 1
            itemnum1 += 1

    for user in User:
        u = user_map[user]
        for item in User[user]:
            i = item_map[item]
            User1[u].append(i)
        Time1[u] = Time[user]

    User = defaultdict(list)
    Time = defaultdict(list)

    f = open('cross_data/processed_data_all/%s_train.csv' % fname2, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        user_ids.append(u)
        item_ids2.append(i)
        User[u].append(i)
        Time[u].append(t)

    f = open('cross_data/processed_data_all/%s_valid.csv' % fname2, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        user_ids.append(u)
        item_ids2.append(i)
        User[u].append(i)
        Time[u].append(t)

    f = open('cross_data/processed_data_all/%s_test.csv' % fname2, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        user_ids.append(u)
        item_ids2.append(i)
        User[u].append(i)
        Time[u].append(t)

    for i in item_ids2:
        if i not in item_map:
            item_map[i] = itemnum2 + 1
            itemnum2 += 1

    for user in User:
        u = user_map[user]
        for item in User[user]:
            i = item_map[item]
            User2[u].append(i)
        Time2[u] = Time[user]

    User = defaultdict(list)
    Time = defaultdict(list)

    f = open('cross_data/processed_data_all/%s_train.csv' % fname3, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        user_ids.append(u)
        item_ids3.append(i)
        User[u].append(i)
        Time[u].append(t)

    f = open('cross_data/processed_data_all/%s_valid.csv' % fname3, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        user_ids.append(u)
        item_ids3.append(i)
        User[u].append(i)
        Time[u].append(t)

    f = open('cross_data/processed_data_all/%s_test.csv' % fname3, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        user_ids.append(u)
        item_ids3.append(i)
        User[u].append(i)
        Time[u].append(t)

    for i in item_ids3:
        if i not in item_map:
            item_map[i] = itemnum3 + 1
            itemnum3 += 1

    for user in User:
        u = user_map[user]
        for item in User[user]:
            i = item_map[item]
            User3[u].append(i)
        Time3[u] = Time[user]

    f = open('cross_data/processed_data_all/%s_negative.csv' % fname, 'r')
    for line in f:
        l = line.rstrip().split(',')
        u = user_map[int(l[0])]
        for j in range(1, 101):
            i = item_map[int(l[j])]
            neglist1[u].append(i)

    for user in User1:
        nfeedback = len(User1[user])
        if nfeedback < 3:
            user_train1[user] = User1[user]
            user_valid1[user] = []
            user_test1[user] = []
        else:
            user_train1[user] = User1[user][:-2]
            user_valid1[user] = []
            user_valid1[user].append(User1[user][-2])
            user_test1[user] = []
            user_test1[user].append(User1[user][-1])
        user_neg1[user] = neglist1[user]

    for user in User2:
        nfeedback = len(User2[user])
        if nfeedback < 3:
            user_train2[user] = User2[user]
            user_valid2[user] = []
            user_test2[user] = []
        else:
            user_train2[user] = User2[user][:-2]
            user_valid2[user] = []
            user_valid2[user].append(User2[user][-2])
            user_test2[user] = []
            user_test2[user].append(User2[user][-1])

    for user in User3:
        nfeedback = len(User3[user])
        if nfeedback < 3:
            user_train3[user] = User3[user]
            user_valid3[user] = []
            user_test3[user] = []
        else:
            user_train3[user] = User3[user][:-2]
            user_valid3[user] = []
            user_valid3[user].append(User3[user][-2])
            user_test3[user] = []
            user_test3[user].append(User3[user][-1])

    return [user_train1, user_valid1, user_test1, usernum, itemnum1, user_neg1, user_train2, user_valid2, user_test2,
            itemnum2, user_train3, user_valid3, user_test3, itemnum3, Time1, Time2, Time3]




def evaluate_next(model,dataset,args,is_test,is_next_one):

    [train, valid, test, usernum, itemnum1, neg, user_train2, user_valid2, user_test2, itemnum2, user_train3,
     user_valid3, user_test3, itemnum3, time1, time2, time3, target_domain_all_items] = dataset

    #评估@k，k=5，10，20
    K=[5,10,20,50]
    NDCG= defaultdict(float)
    Recall= defaultdict(float)
    best_dcg=defaultdict(float)
    evaluate_user_num=0
    evaluate_item_num=0
    # num=0



    users = range(1, usernum + 1)
    for u in users:
        # 根据验证还是测试，拼接测试序列
        if is_test:
            train1_u = train[u] + valid[u]
            train2_u = user_train2[u] + user_valid2[u]
            train3_u = user_train3[u] + user_valid3[u]
            test_u=test[u]
        else:
            train1_u = copy.deepcopy(train[u])
            train2_u = copy.deepcopy(user_train2[u])
            train3_u = copy.deepcopy(user_train3[u])
            test_u = copy.deepcopy(valid[u])

        test_item_num = len(test_u)
        evaluate_item_num+=test_item_num

        # if len(train1_u)==1: num+=1
        if len(train1_u) < 2 or len(test_u) < 1: continue

        # 将序列补充到固定长度

        seq = np.zeros([args.maxlen], dtype=np.int32)
        t1 = np.zeros([args.maxlen], dtype=np.int32)
        t2 = np.zeros([args.maxlen], dtype=np.int32)
        # t3 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        for i, t in reversed(list(zip(train1_u, time1[u]))):
            seq[idx] = i
            t1[idx] = t
            idx -= 1
            if idx == -1: break
        rated = set(train1_u)
        rated.add(0)
        if is_next_one:
            item_idx = [test_u[0]]
        else:
            item_idx = list(copy.deepcopy(test_u))

        seq2 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(train2_u, time2[u]))):
            seq2[idx] = i
            t2[idx] = t
            idx -= 1
            if idx == -1: break

        mask2 = np.zeros([args.maxlen], dtype=np.int32)
        idx2 = 0
        for idx in range(len(seq)):
            while idx2 < args.maxlen and t1[idx] >= t2[idx2]:
                idx2 += 1
            mask2[idx] = idx2

        # seq3 = np.zeros([args.maxlen], dtype=np.int32)
        # idx = args.maxlen - 1
        # for i, t in reversed(list(zip(train3_u, time3[u]))):
        #     seq3[idx] = i
        #     t3[idx] = t
        #     idx -= 1
        #     if idx == -1: break
        #
        # mask3 = np.zeros([args.maxlen], dtype=np.int32)
        # idx3 = 0
        # for idx in range(len(seq)):
        #     while idx3 < args.maxlen and t1[idx] >= t3[idx3]:
        #         idx3 += 1
        #     mask3[idx] = idx3


        # if args.is_full_candidate:
        # if False:
        #     # 分批次预测
        #     batch_size=1000
        #     batch_predictions = []
        #     num_batches=len(neg[u])//batch_size + (1 if len(neg[u]) % batch_size!= 0 else 0)
        #     for batch_idx in range(num_batches):
        #         if batch_idx == 0:
        #             batch_item_idx=[test_u[0]]
        #         else:
        #             batch_item_idx=[]
        #
        #         start_idx = batch_idx * batch_size
        #         end_idx = min((batch_idx + 1) * batch_size, len(neg[u]))
        #
        #         batch_item_idx =batch_item_idx +neg[u][start_idx:end_idx]
        #
        #         predictions = -model.predict(
        #             *[np.array(l) for l in [[u], [seq], [seq2], [seq3], batch_item_idx, [mask2], [mask3]]])
        #         predictions = predictions[0]
        #
        #         batch_predictions.append(predictions)
        #
        #     all_predictions = torch.cat(batch_predictions, dim=0)
        #     rank = all_predictions.argsort().argsort()[0].item()
        # else:
        for i in neg[u]:
            item_idx.append(i)

        if args.model=='sasrec':
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        elif args.model=='tjapl':
            # predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [seq2], [seq3], item_idx, [mask2], [mask3]]])
            predictions = -model.predict(
                *[np.array(l) for l in [[u], [seq], [seq2],  item_idx, [mask2]]])

        predictions = predictions[0]



        if not is_next_one:
            rank = [predictions.argsort().argsort()[i].item() for i in range(test_item_num)]
            rec_u=defaultdict(float)
            dcg_u=defaultdict(float)
            for k in K:
                for idx in range(test_item_num):
                    if rank[idx] < k:
                        dcg_u[k] += 1 / np.log2(rank[idx] + 2)
                        rec_u[k] += 1

                if dcg_u[k] >= best_dcg[k]: best_dcg[k] = dcg_u[k]
                Recall[k]+=rec_u[k] / test_item_num
                NDCG[k]+=dcg_u[k]
        else:
            rank = predictions.argsort().argsort()[0].item()
            for k in K:
                if rank < k:
                    NDCG[k] += 1 / np.log2(rank + 2)
                    Recall[k] += 1

        evaluate_user_num += 1

            # if evaluate_user_num % 100 == 0:
            #     print('.', end="")
            #     sys.stdout.flush()
    # print("\nis_test",is_test,end='')
    ndcg={}
    rec={}
    for k in K:
        if is_next_one:
            ndcg[k]=NDCG[k]/evaluate_user_num
            rec[k]=Recall[k]/evaluate_user_num
        else:
            ndcg[k] = NDCG[k] / (evaluate_user_num*best_dcg[k])
            rec[k] = Recall[k] / evaluate_user_num
        print('NDCG@%d: %.6f,Rec@%d: %.6f' %(k,ndcg[k], k,rec[k]),end="; ")

    print('evaluate_user_num',evaluate_user_num,"evaluate_item_num",evaluate_item_num)

    return ndcg,rec









def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum1, neg, user_train2, user_valid2, user_test2, itemnum2, user_train3,
     user_valid3, user_test3, itemnum3, time1, time2, time3,target_domain_all_items] = copy.deepcopy(dataset)

    valid_user = 0.0
    NDCG_10 = 0.0
    HT_10 = 0.0

    users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        t1 = np.zeros([args.maxlen], dtype=np.int32)
        t2 = np.zeros([args.maxlen], dtype=np.int32)
        t3 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i, t in reversed(list(zip(train[u], time1[u]))):
            seq[idx] = i
            t1[idx] = t
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]

        seq2 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(user_train2[u], time2[u]))):
            seq2[idx] = i
            t2[idx] = t
            idx -= 1
            if idx == -1: break

        mask2 = np.zeros([args.maxlen], dtype=np.int32)
        idx2 = 0
        for idx in range(len(seq)):
            while idx2 < args.maxlen and t1[idx] >= t2[idx2]:
                idx2 += 1
            mask2[idx] = idx2

        seq3 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(user_train3[u], time3[u]))):
            seq3[idx] = i
            t3[idx] = t
            idx -= 1
            if idx == -1: break

        mask3 = np.zeros([args.maxlen], dtype=np.int32)
        idx3 = 0
        for idx in range(len(seq)):
            while idx3 < args.maxlen and t1[idx] >= t3[idx3]:
                idx3 += 1
            mask3[idx] = idx3

        for i in neg[u]:
            item_idx.append(i)

        predictions = -model.predict(
            *[np.array(l) for l in [[u], [seq], [seq2], [seq3], item_idx, [mask2], [mask3]]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG_10 / valid_user, HT_10 / valid_user


def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum1, neg, user_train2, user_valid2, user_test2, itemnum2, user_train3,
     user_valid3, user_test3, itemnum3, time1, time2, time3] = copy.deepcopy(dataset)

    valid_user = 0.0
    NDCG_10 = 0.0
    HT_10 = 0.0
    users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        t1 = np.zeros([args.maxlen], dtype=np.int32)  #
        t2 = np.zeros([args.maxlen], dtype=np.int32)  #
        t3 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(train[u], time1[u]))):
            seq[idx] = i
            t1[idx] = t
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]

        seq2 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(user_train2[u], time2[u]))):
            seq2[idx] = i
            t2[idx] = t
            idx -= 1
            if idx == -1: break

        mask2 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx2 = 0
        for idx in range(len(seq)):
            while idx2 < args.maxlen and t1[idx] >= t2[idx2]:
                idx2 += 1
            mask2[idx] = idx2

        seq3 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(user_train3[u], time3[u]))):
            seq3[idx] = i
            t3[idx] = t
            idx -= 1
            if idx == -1: break

        mask3 = np.zeros([args.maxlen], dtype=np.int32)
        idx3 = 0
        for idx in range(len(seq)):
            while idx3 < args.maxlen and t1[idx] >= t3[idx3]:
                idx3 += 1
            mask3[idx] = idx3

        for i in neg[u]:
            item_idx.append(i)

        predictions = -model.predict(
            *[np.array(l) for l in [[u], [seq], [seq2], [seq3], item_idx, [mask2], [mask3]]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG_10 / valid_user, HT_10 / valid_user
