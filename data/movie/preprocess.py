import numpy as np
import argparse
import pandas as pd


def read_rating():
    users, items, timestamps = [], [], []

    with open('./ratings.dat', 'r') as f:
        for line in f.readlines():
            splited = line.split('::')
            user, item, rating, timestamp = int(splited[0]), int(splited[1]), int(splited[2]), int(splited[3])

            # 如果启用了threshold且评分小于threshold才过滤，其余情况都是直接放入
            if args.use_threshold and rating < args.threshold:
                continue

            users.append(user)
            items.append(item)
            timestamps.append(timestamp)

    df = pd.DataFrame({
        'user': users,
        'item': items,
        'timestamp': timestamps
    })

    df.to_csv('./ratings.csv', index=False)


def remove_less_than_5():
    df = pd.read_csv('./ratings.csv', dtype={'user': int, 'item': int, 'timestamp': int})

    # 先根据user分组，找出少于五条交互记录的用户id
    grouped = df.groupby(by='user')
    for k, v in grouped:
        if len(v) < 5:
            remove_user.append(k)

    # 再根据item分组，找出少于五条交互记录的物品id
    grouped = df.groupby(by='item')
    for k, v in grouped:
        if len(v) < 5:
            remove_item.append(k)

    # 那些要被过滤掉的行为True
    mask = (df['user'].isin(remove_user)) | (df['item'].isin(remove_item))
    new_df = df.loc[~mask]

    new_df.to_csv('processed_ratings.csv', index=False)


def convert_id(remove_item):
    # entity的idx从1开始(padding)
    user_cnt, entity_cnt = 0, 1
    with open('./item_index2entity_id_rehashed.txt', 'r') as f:
        for line in f.readlines():
            splited = line.split('\t')
            item_idx, entity_idx = int(splited[0]), int(splited[1])
            # 被排除掉的物品不能计算在内
            if item_idx not in remove_item:
                item_idx_old2new[item_idx] = entity_cnt
                entity_idx_old2new[entity_idx] = entity_cnt
                entity_cnt += 1

            if item_idx in remove_item:
                remove_entity.append(entity_idx)

    users, items, timestamps = [], [], []

    df = pd.read_csv('./processed_ratings.csv', dtype={'user': int, 'item': int, 'timestamp': int})
    for idx, row in df.iterrows():
        user_old, item_old, timestamp = row['user'], row['item'], row['timestamp']

        # 如果用户交互过的物品没有出现在知识图谱中，武略这个交互记录
        if item_old not in item_idx_old2new.keys():
            continue

        if user_old not in user_idx_old2new.keys():
            user_idx_old2new[user_old] = user_cnt
            user_cnt += 1

        user = user_idx_old2new[user_old]
        item = item_idx_old2new[item_old]

        users.append(user)
        items.append(item)
        timestamps.append(timestamp)

    new_df = pd.DataFrame({
        'user': users,
        'item': items,
        'timestamp': timestamps
    })

    sorted_df = None
    grouped = new_df.groupby(by='user')

    for k, v in grouped:
        sorted_v = v.sort_values(by='timestamp')
        if sorted_df is None:
            sorted_df = sorted_v
        else:
            sorted_df = pd.concat([sorted_df, sorted_v], axis=0)

    sorted_df.to_csv(f'./final_ratings.csv', index=False)

    print(f'number of users: {len(user_idx_old2new)}')
    print(f'number of items: {len(item_idx_old2new)}')


def convert_kg():
    entity_cnt = max(list(entity_idx_old2new.values())) + 1
    relation_cnt = 1

    files = []
    files.append(open('./kg_part1_rehashed.txt'))
    files.append(open('./kg_part2_rehashed.txt'))
    # files.append(open('./kg_rehashed.txt'))
    writer = open('./kg_final.txt', 'w', encoding='utf-8')

    for f in files:
        for line in f.readlines():
            splited = line.strip().split('\t')
            head_old, relation_old, tail_old = int(splited[0]), splited[1], int(splited[2])

            if head_old in remove_entity or tail_old in remove_entity:
                continue

            if head_old not in entity_idx_old2new.keys():
                entity_idx_old2new[head_old] = entity_cnt
                entity_cnt += 1
            head = entity_idx_old2new[head_old]

            if relation_old not in relation2idx.keys():
                relation2idx[relation_old] = relation_cnt
                relation_cnt += 1
            relation = relation2idx[relation_old]

            if tail_old not in entity_idx_old2new.keys():
                entity_idx_old2new[tail_old] = entity_cnt
                entity_cnt += 1
            tail = entity_idx_old2new[tail_old]

            writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print(f'number of entities(contain items and padding: 0): {entity_cnt + 1}')
    print(f'number of relations: {relation_cnt}')


if __name__ == '__main__':
    np.random.seed(2020)

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_threshold', type=bool, default=False, help='False means user all data as positive data')
    parser.add_argument('--threshold', type=int, default=4, help='greater than or equal threshold')

    args = parser.parse_args()

    remove_user, remove_item = [], []
    # remove_item对应的entity
    remove_entity = []

    relation2idx = dict()
    entity_idx_old2new = dict()
    item_idx_old2new = dict()
    user_idx_old2new = dict()

    read_rating()
    remove_less_than_5()
    convert_id(remove_item)
    convert_kg()
