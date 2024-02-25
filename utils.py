import os.path
import pickle
import random

import pandas as pd
import numpy as np
from collections import defaultdict

import torch
from tqdm import tqdm
from multiprocessing import Process, Queue


def load_data(args, pretrained_model):
    user_train, user_valid, user_test, user_all, user_nums, item_nums = data_partition(args)
    entity_nums, relation_nums, kg = load_kg(args)

    ripple_set_file = './data/' + args.dataset + f'/ripple_set_{args.n_memory}.dict'
    if os.path.exists(ripple_set_file):
        with open(ripple_set_file, 'rb') as f:
            ripple_set = pickle.load(f)
    else:
        if args.use_pretrained:
            ripple_set = get_ripple_set_by_similarity(args, kg, user_all, ripple_set_file, pretrained_model)
        else:
            ripple_set = get_ripple_set(args, kg, user_all, ripple_set_file)

    return user_train, user_valid, user_test, user_nums, item_nums, entity_nums, relation_nums, ripple_set


# 在划分数据的时候就将数据切割成合适的长度
def data_partition(args):
    print('spliting the sequence data into the train/valid/test...')

    file = './data/' + args.dataset + '/final_ratings.csv'

    user_train, user_valid, user_test, user_all = {}, {}, {}, {}

    df = pd.read_csv(file)
    grouped = df.groupby(by='user')

    max_user_id, max_item_id = -1, -1
    for user, user_df in tqdm(grouped):
        max_user_id = max(user, max_user_id)
        max_item_id = max(max_item_id, user_df['item'].max())

        user_items = list(user_df['item'].values)
        if len(user_items) < 3:
            user_train[user] = user_items
            user_valid[user] = []
            user_test[user] = []

        else:
            user_train[user] = user_items[:-2]
            user_valid[user] = [user_items[-2]]
            user_test[user] = [user_items[-1]]

        user_all[user] = user_items

    return user_train, user_valid, user_test, user_all, max_user_id + 1, max_item_id + 1


def construct_kg(kg_data):
    print('constructing knowledge graph...')

    kg = defaultdict(list)
    for i in tqdm(range(kg_data.shape[0])):
        head, relation, tail = kg_data[i, :]
        kg[head].append((tail, relation))

    return kg


def load_kg(args):
    file = './data/' + args.dataset + '/kg_final.txt'

    kg_data = np.loadtxt(file, dtype=int)

    max_entity_id = max(set(kg_data[:, 0]) | set(kg_data[:, 2]))
    max_relation_id = max(kg_data[:, 1])

    kg = construct_kg(kg_data)

    return max_entity_id + 1, max_relation_id + 1, kg


# 对于每个序列的每个物品，找到其Ripple set
# 需要注意的，找到的ripple set，其不能出现交互时间晚于该物品的其余物品

def get_ripple_set(args, kg, user_all, ripple_set_file):
    np.random.seed(2024)
    print('getting ripple set...')

    # 加入user键而不是使用全局的ripple set是因为在进行nhop可能会有因为用户不同而导致ripple set的情况
    # 主要的原因在上一个注释中，每个用户对每个物品的交互时间是不同的

    # user -> {item1: [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails)...],
    #          item2: [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails)...]}
    ripple_set = defaultdict(dict)

    for user in tqdm(user_all.keys()):
        for item in user_all[user]:
            for hop in range(args.n_hop):
                memories_h, memories_r, memories_t = [], [], []

                if hop == 0:
                    tails_of_last_hop = [item]
                else:
                    tails_of_last_hop = ripple_set[user][item][-1][2]

                # 这里需要注前面提到的点
                for entity in tails_of_last_hop:
                    for tail_and_relation in kg[entity]:
                        memories_h.append(entity)
                        memories_r.append(tail_and_relation[1])
                        memories_t.append(tail_and_relation[0])

                # 有可能会出现没有下一跳的情况, 且第0跳不会发生这种情况
                # 解决方案：1.下一跳全部使用padding来代替
                #         2.利用源码的思路，复制上一跳的ripple set   看看哪种效果更好
                if len(memories_h) == 0:
                    if hop == 0:
                        memories_h = [0 for _ in range(args.n_memory)]
                        memories_r = [0 for _ in range(args.n_memory)]
                        memories_t = [0 for _ in range(args.n_memory)]
                    else:
                        last_ripple_set = ripple_set[user][item][-1]
                        memories_h = last_ripple_set[0]
                        memories_r = last_ripple_set[1]
                        memories_t = last_ripple_set[2]
                else:
                    # 裁剪
                    replace = len(memories_h) < args.n_memory
                    indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
                    memories_h = [memories_h[i] for i in indices]
                    memories_r = [memories_r[i] for i in indices]
                    memories_t = [memories_t[i] for i in indices]

                if item not in ripple_set[user].keys():
                    ripple_set[user][item] = []
                ripple_set[user][item].append((memories_h, memories_r, memories_t))

    with open(ripple_set_file, 'wb') as f:
        pickle.dump(ripple_set, f, protocol=pickle.HIGHEST_PROTOCOL)

    return ripple_set


# 利用训练好的模型根据相似度进行采样，而不是随机采样
def get_ripple_set_by_similarity(args, kg, user_all, ripple_set_file, pretrained_model):
    # 不需要设置随机种子
    print('getting ripple set...')

    ripple_set = defaultdict(dict)

    for user in tqdm(user_all.keys()):
        for item in user_all[user]:
            for hop in range(args.n_hop):
                memories_h, memories_r, memories_t = [], [], []

                if hop == 0:
                    tails_of_last_hop = [item]
                else:
                    tails_of_last_hop = ripple_set[user][item][-1][2]

                for entity in tails_of_last_hop:
                    for tail_and_relation in kg[entity]:
                        memories_h.append(entity)
                        memories_r.append(tail_and_relation[1])
                        memories_t.append(tail_and_relation[0])

                if len(memories_h) == 0:
                    if args.use_ripple_padding:
                        memories_h = [0 for _ in range(args.n_memory)]
                        memories_r = [0 for _ in range(args.n_memory)]
                        memories_t = [0 for _ in range(args.n_memory)]
                    else:
                        last_ripple_set = ripple_set[user][item][-1]
                        memories_h = last_ripple_set[0]
                        memories_r = last_ripple_set[1]
                        memories_t = last_ripple_set[2]
                else:

                    replace = len(memories_h) <= args.n_memory
                    # 如果不满足32个，则进行和之前一样np.sample(采不采样都一样，都是全取)
                    if replace:
                        indices = np.random.choice(len(memories_h), size=args.n_memory, replace=True)
                        memories_h = [memories_h[i] for i in indices]
                        memories_r = [memories_r[i] for i in indices]
                        memories_t = [memories_t[i] for i in indices]
                    # 超过了，则根据相似度获取最优的
                    else:
                        # 根据相似度进行筛选
                        tensor_h = torch.tensor(memories_h, dtype=torch.int64, device=args.device)
                        tensor_r = torch.tensor(memories_r, dtype=torch.int64, device=args.device)
                        tensor_t = torch.tensor(memories_t, dtype=torch.int64, device=args.device)

                        h_embed, r_embed, t_embed = pretrained_model.entity_embed(tensor_h), pretrained_model.entity_embed(
                            tensor_r), pretrained_model.entity_embed(tensor_t)
                        W_r = pretrained_model.trans_M(tensor_r).view(-1, args.hidden_units, args.hidden_units)

                        trans_h = torch.squeeze(torch.bmm(torch.unsqueeze(h_embed, dim=1), W_r), dim=1)
                        trans_t = torch.squeeze(torch.bmm(torch.unsqueeze(t_embed, dim=1), W_r), dim=1)
                        # 距离越小越好，代表相似度越高
                        d = torch.norm(trans_h + r_embed - trans_t, dim=1)
                        # 排序，取最小的n_memory个，即indices里的前n_memory个元素
                        indices = torch.argsort(d)[:args.n_memory].tolist()
                        memories_h = [memories_h[indice] for indice in indices]
                        memories_r = [memories_r[indice] for indice in indices]
                        memories_t = [memories_t[indice] for indice in indices]

                    if item not in ripple_set[user].keys():
                        ripple_set[user][item] = []
                    ripple_set[user][item].append((memories_h, memories_r, memories_t))

    with open(ripple_set_file, 'wb') as f:
        pickle.dump(ripple_set, f, protocol=pickle.HIGHEST_PROTOCOL)

    return ripple_set


def random_neg(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)

    return t


def sample_function(user_train, ripple_set, user_nums, item_nums, batch_size, maxlen, n_hop, n_memory, result_queue,
                    SEED):
    def sample():
        user = np.random.randint(1, user_nums)

        while len(user_train[user]) <= 1:
            user = np.random.randint(1, user_nums)

        # input
        seq = np.zeros([maxlen], dtype=int)
        # label
        pos = np.zeros([maxlen], dtype=int)
        neg = np.zeros([maxlen], dtype=int)
        nxt = user_train[user][-1]
        idx = maxlen - 1
        #
        seq_ripple_set = np.zeros([maxlen, n_hop, 3, n_memory], dtype=int)

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neg(1, item_nums, ts)

            seq_ripple_set[idx] = np.array(ripple_set[user][i])
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return user, seq, pos, neg, seq_ripple_set

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, ripple_set, user_nums, item_nums, batch_size, maxlen, n_hop, n_memory, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(
                    User, ripple_set, user_nums, item_nums, batch_size, maxlen, n_hop, n_memory, self.result_queue,
                    np.random.randint(2e9)))
            )

        self.processors[-1].daemon = True
        self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# 验证集
def evaluate_valid(model, train, valid, user_nums, item_nums, ripple_set, args):
    NDCG = 0.0
    HIT = 0.0
    valid_user = 0

    if user_nums > 10000:
        users = random.sample(range(0, user_nums), 10000)
    else:
        users = range(0, user_nums)

    for user in users:

        if len(train[user]) < 1 or len(valid[user]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=int)
        seq_ripple_set = np.zeros([args.maxlen, args.n_hop, 3, args.n_memory], dtype=int)
        idx = args.maxlen - 1

        for i in reversed(train[user]):
            seq[idx] = i
            seq_ripple_set[idx] = np.array(ripple_set[user][i])
            idx -= 1
            if idx == -1:
                break

        rated = set(train[user])
        item_indices = [valid[user][0]]
        for _ in range(100):
            t = np.random.randint(1, item_nums)
            while t in rated:
                t = np.random.randint(1, item_nums)
            item_indices.append(t)

        memory_h = seq_ripple_set[:, :, 0, :]
        memory_r = seq_ripple_set[:, :, 1, :]
        memory_t = seq_ripple_set[:, :, 2, :]

        # 转换成batch形式的
        user = torch.unsqueeze(torch.tensor(user, dtype=torch.int64).to(args.device), dim=0)
        seq = torch.unsqueeze(torch.tensor(seq, dtype=torch.int64).to(args.device), dim=0)
        item_indices = torch.tensor(item_indices, dtype=torch.int64).to(args.device)
        memory_h = torch.unsqueeze(torch.tensor(memory_h, dtype=torch.int64).to(args.device), dim=0)
        memory_r = torch.unsqueeze(torch.tensor(memory_r, dtype=torch.int64).to(args.device), dim=0)
        memory_t = torch.unsqueeze(torch.tensor(memory_t, dtype=torch.int64).to(args.device), dim=0)

        predictions = -model.predict(user, seq, item_indices, memory_h, memory_r, memory_t)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1

    return NDCG / valid_user, HIT / valid_user


# 测试集
def evaluate(model, train, valid, test, user_nums, item_nums, ripple_set, args):
    NDCG = 0.0
    HIT = 0.0
    valid_user = 0

    if user_nums > 10000:
        users = random.sample(range(1, user_nums), 10000)
    else:
        users = range(1, user_nums)

    for user in users:

        if len(train[user]) < 1 or len(test[user]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=int)
        seq_ripple_set = np.zeros([args.maxlen, args.n_hop, 3, args.n_memory], dtype=int)
        idx = args.maxlen - 1
        seq[idx] = valid[user][0]
        idx -= 1

        for i in reversed(train[user]):
            seq[idx] = i
            seq_ripple_set[idx] = np.array(ripple_set[user][i])
            idx -= 1
            if idx == -1:
                break

        rated = set(train[user])
        item_indices = [test[user][0]]
        for _ in range(100):
            t = np.random.randint(1, item_nums)
            while t in rated:
                t = np.random.randint(1, item_nums)
            item_indices.append(t)

        memory_h = seq_ripple_set[:, :, 0, :]
        memory_r = seq_ripple_set[:, :, 1, :]
        memory_t = seq_ripple_set[:, :, 2, :]

        # 转换成batch形式的
        user = torch.unsqueeze(torch.tensor(user, dtype=torch.int64).to(args.device), dim=0)
        seq = torch.unsqueeze(torch.tensor(seq, dtype=torch.int64).to(args.device), dim=0)
        item_indices = torch.tensor(item_indices, dtype=torch.int64).to(args.device)
        memory_h = torch.unsqueeze(torch.tensor(memory_h, dtype=torch.int64).to(args.device), dim=0)
        memory_r = torch.unsqueeze(torch.tensor(memory_r, dtype=torch.int64).to(args.device), dim=0)
        memory_t = torch.unsqueeze(torch.tensor(memory_t, dtype=torch.int64).to(args.device), dim=0)

        predictions = -model.predict(user, seq, item_indices, memory_h, memory_r, memory_t)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1

    return NDCG / valid_user, HIT / valid_user
