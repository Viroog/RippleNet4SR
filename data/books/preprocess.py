import json
import pandas as pd


# 原始数据本身就有序
def read_rating():
    users, items = [], []

    with open('sequential_data.txt', 'r') as f:
        for line in f.readlines():
            splited = line.strip().split(' ')
            user, item_ids = splited[0], splited[1:]

            for item in item_ids:
                users.append(int(user))
                items.append(int(item))

    assert len(users) == len(items)

    df = pd.DataFrame({
        'user': users,
        'item': items
    })

    df.to_csv('rating.csv', index=False)


def remove_less_than_5():
    df = pd.read_csv('rating.csv', dtype={'user': int, 'item': int})

    # 先根据user分组
    grouped = df.groupby(by='user')
    for k, v in grouped:
        if len(v) < 5:
            removed_user.append(k)

    # 再根据item分组
    grouped = df.groupby(by='item')
    for k, v in grouped:
        if len(v) < 5:
            removed_item.append(k)

    mask = (df['user'].isin(removed_user)) | (df['item'].isin(removed_item))
    new_df = df.loc[~mask]

    new_df.to_csv('processed_ratings.csv', index_label=False)


# 将用户，物品以及kg实体重新映射
def convert_id():
    # 0是给padding用的
    entity_cnt = 1

    with open('datamaps.json', 'r') as f:
        datamaps = json.load(f)

    with open('entity2text.json', 'r') as f:
        entity2text = json.load(f)

    with open('item2entity.json', 'r') as f:
        item2entity = json.load(f)

    item_new_ids, origin_ids, freebase_ids, texts = [], [], [], []
    # 只要保证item_old_id和item_new_id为int即可，其他都包含了字符
    for item_old_id, origin_id in datamaps['id2item'].items():
        item_old_id, origin_id = int(item_old_id), origin_id

        if item_old_id not in removed_item:
            item_old2new[item_old_id] = entity_cnt

            item_new_ids.append(item_old2new[item_old_id])
            origin_ids.append(origin_id)
            freebase_id = item2entity[origin_id]
            freebase_ids.append(freebase_id)
            text = entity2text[freebase_id]
            texts.append(text)

            freebase2id[freebase_id] = entity_cnt
            entity_cnt += 1

    df = pd.DataFrame({
        'item_new_id': item_new_ids,
        'origin_id': origin_ids,
        'freebase_id': freebase_ids,
        'text': texts
    })

    df = df.sort_values(by='item_new_id')
    df.to_csv('mapping.csv', index=False)

    user_cnt = 0
    users, items = [], []
    # 交互数据，用于重新映射用户
    interact_df = pd.read_csv('processed_ratings.csv')
    for idx, row in interact_df.iterrows():
        user_old, item_old = int(row[0]), int(row[1])

        if user_old not in user_old2new:
            user_old2new[user_old] = user_cnt
            user_cnt += 1
        user_new = user_old2new[user_old]

        item_new = item_old2new[item_old]

        users.append(user_new)
        items.append(item_new)

    final_df = pd.DataFrame({
        'user': users,
        'item': items
    })

    final_df.to_csv('final_ratings.csv', index=False)

    print(f'user nums: {len(user_old2new)}')
    print(f'item nums: {len(item_old2new)}')
    print(f'sparsity: {(len(final_df) / (len(user_old2new) * len(item_old2new))) * 100}%')


def convert_kg():
    entity_cnt = max(freebase2id.values()) + 1
    relation_cnt = 1

    with open('triple2dict.json', 'r') as f:
        kg_dict = json.load(f)

    writer = open('kg_final.txt', 'w', encoding='utf-8')

    for head, relation_tail_dict in kg_dict.items():
        if head not in freebase2id.keys():
            freebase2id[head] = entity_cnt
            entity_cnt += 1

        head_id = freebase2id[head]

        for relation, tail_list in relation_tail_dict.items():
            if relation not in relation2id.keys():
                relation2id[relation] = relation_cnt
                relation_cnt += 1

            relation_id = relation2id[relation]

            for tail in tail_list:
                if tail not in freebase2id.keys():
                    freebase2id[tail] = entity_cnt
                    entity_cnt += 1

                tail_id = freebase2id[tail]

                writer.write('%d\t%d\t%d\n' % (head_id, relation_id, tail_id))

    writer.close()
    print(f'entity nums: {len(freebase2id) + 1}')
    print(f'relation nums: {len(relation2id)}')

if __name__ == '__main__':

    removed_user, removed_item = [], []

    read_rating()
    remove_less_than_5()

    # 结果表明，amazon books数据集没有需要去除的用户和物品

    freebase2id = {}
    item_old2new = {}
    user_old2new = {}
    relation2id = {}

    convert_id()
    convert_kg()
