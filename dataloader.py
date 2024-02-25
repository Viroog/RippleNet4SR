from torch.utils.data import dataset
import pandas as pd
from collections import defaultdict
import numpy as np


class DataLoader(dataset.Dataset):
    def __init__(self, entity_nums ,args):
        super(DataLoader, self).__init__()

        self.entity_nums = entity_nums

        kg_file_path = f'./data/movie/{args.kg_file}'
        self.kg = pd.read_csv(kg_file_path)

        self.h_list = self.kg['h'].values
        self.r_list = self.kg['r'].values
        self.pos_t_list = self.kg['t'].values

        self.kg_dict = self.get_kg_dict()

    def get_kg_dict(self):
        kg_dict = defaultdict(list)

        for row in self.kg.iterrows():
            h, r, t = row[1]
            kg_dict[h].append((r, t))

        return kg_dict

    # 随机替换掉一个尾节点
    def sample_one_neg_tail(self, h, r):
        pos_triple = self.kg_dict[h]

        while True:
            neg_t = np.random.randint(low=1, high=self.entity_nums, size=1)[0]

            if (r, neg_t) not in pos_triple:
                return neg_t

    def __getitem__(self, idx):
        h, r, pos_t = self.h_list[idx], self.r_list[idx], self.pos_t_list[idx]

        neg_t = self.sample_one_neg_tail(h, r)

        return h, r, pos_t, neg_t

    def __len__(self):
        return len(self.kg_dict)
