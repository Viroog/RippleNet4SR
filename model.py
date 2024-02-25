import torch
import torch.nn as nn
import torch.nn.functional as F


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs = outputs + inputs
        return outputs


class RippleNet4SR(nn.Module):
    def __init__(self, user_nums, entity_nums, relation_nums, pretrained_model, args):
        super(RippleNet4SR, self).__init__()

        # 给时刻0的前一个状态的输入(即理解为用户的本质特征)
        self.user_embed = nn.Embedding(user_nums, args.hidden_units)
        # 包括了物品+知识图谱中的所有实体
        self.entity_embed = nn.Embedding(entity_nums, args.hidden_units, padding_idx=0)
        # 两种方式
        self.relation_embed = nn.Embedding(relation_nums, args.hidden_units * args.hidden_units, padding_idx=0)
        # self.relation_embed = nn.Parameter(torch.randn(relation_nums, args.hidden_units, args.hidden_units))
        # item_embed更新时用到的linear transform
        self.transform_matrix = nn.Linear(args.hidden_units, args.hidden_units, bias=False)
        self.interest_transform_matrix_1 = nn.Linear(args.hidden_units, args.hidden_units, bias=False)
        self.interest_transform_matrix_2 = nn.Linear(args.hidden_units, args.hidden_units, bias=False)
        # self.interest_transform_matrix = nn.Linear(args.hidden_units * 2, args.hidden_units, bias=False)

        self.pos_embed = nn.Embedding(args.maxlen, args.hidden_units)
        self.embed_dropout = nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.block_nums):
            new_attn_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(args.hidden_units, args.head_nums, args.dropout_rate,
                                                   batch_first=True)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        self.hidden_units = args.hidden_units
        self.n_memory = args.n_memory
        self.n_hop = args.n_hop
        self.maxlen = args.maxlen
        self.item_update_mode = args.item_update_mode
        self.device = args.device

        self._init_param(args, pretrained_model)

    def _init_param(self, args, pretrained_model):
        for name, param in self.named_parameters():
            try:
                nn.init.xavier_normal_(param.data)
            except:
                pass

        # 加载预训练好的embedding
        if args.use_pretrained:
            self.entity_embed.weight.data = pretrained_model.entity_embed.weight.data
            # self.relation_embed.weight.data = pretrained_model.trans_M.weight.data

    def _update_item_embed(self, item_embed, o):
        if self.item_update_mode == 'replace':
            item_embed = o
        elif self.item_update_mode == 'plus':
            item_embed = o + item_embed
        elif self.item_update_mode == 'replace_transform':
            item_embed = self.transform_matrix(o)
        elif self.item_update_mode == 'plus_transform':
            item_embed = self.transform_matrix(o + item_embed)
        else:
            raise NotImplementedError(f"{self.item_update_mode} item update mode is not yet implement")

        return F.leaky_relu(item_embed)

    # h/t_embed_list: [(batch_size, maxlen, n_memory, hidden_units)]
    # r_embed_list: [(batch_size, maxlen, n_memory, hidden_units, hidden_units)]
    def _key_address(self, h_embed_list, r_embed_list, t_embed_list, item_embed):
        o_list = []
        for hop in range(self.n_hop):
            # (batch_size, maxlen, n_memory, hidden_units, 1)
            h_expanded = torch.unsqueeze(h_embed_list[hop], dim=-1)

            # (batch_size, maxlen, n_memory, hidden_units)
            Rh = torch.squeeze(torch.matmul(r_embed_list[hop], h_expanded), dim=-1)

            # (batch_size, maxlen, hidden_units, 1)
            v = torch.unsqueeze(item_embed, dim=-1)

            # (batch_size, maxlen, n_memory)
            probs = torch.squeeze(torch.matmul(Rh, v), dim=-1)
            # 转换成概率(在最后一个维度上，最后一个维度为n_memory，对32个entity进行概率化)
            probs = F.softmax(probs, dim=-1)

            # (batch_size, maxlen, n_memory, 1)
            probs_expanded = torch.unsqueeze(probs, dim=-1)

            # (batch_size, maxlen, hidden_units)  尾节点的权重和
            o = torch.sum(probs_expanded * t_embed_list[hop], dim=2)

            item_embed = self._update_item_embed(item_embed, o)
            o_list.append(o)

        return o_list, item_embed

    # item: (batch_size, maxlen)
    # memory_x: (batch_size, maxlen ,n_hop, n_memory)
    def get_user_short_interest(self, item, memory_h, memory_r, memory_t):
        # (batch_size, maxlen, hidden_units)
        item_embed = self.entity_embed(item)
        h_embed_list, r_embed_list, t_embed_list = [], [], []

        for hop in range(self.n_hop):
            h_embed_list.append(self.entity_embed(memory_h[:, :, hop, :]))
            r_embed_list.append(
                self.relation_embed(memory_r[:, :, hop, :]).view(-1, self.maxlen, self.n_memory, self.hidden_units,
                                                                 self.hidden_units))
            t_embed_list.append(self.entity_embed(memory_t[:, :, hop, :]))

        o_list, updated_item_embed = self._key_address(h_embed_list, r_embed_list, t_embed_list, item_embed)

        user_short_interest = item_embed
        for hop in range(self.n_hop):
            if user_short_interest is None:
                user_short_interest = o_list[hop]
            else:
                user_short_interest = user_short_interest + o_list[hop]

        return user_short_interest, updated_item_embed
        # return user_short_interest, updated_item_embed, h_embed_list, r_embed_list, t_embed_list

    # short interest + long interest
    def get_user_interest(self, user, user_short_interest):
        # user_interest: (batch_size, maxlen, hidden_units)
        user_interest = None

        for i in range(self.maxlen):
            if i == 0:
                user_long_interest_t = self.user_embed(user)
            else:
                # 最后一个为用户上一个状态的兴趣
                user_long_interest_t = user_interest[:, -1, :]

            user_short_interest_t = user_short_interest[:, i, :]

            # user_interest_t = torch.concat([user_short_interest_t, user_long_interest_t], dim=1)
            user_interest_t_1 = user_short_interest_t + user_long_interest_t
            user_interest_t_2 = user_short_interest_t * user_long_interest_t

            # transform_user_interest_t = F.relu(self.interest_transform_matrix(user_interest_t))
            # transform_user_interest_t = F.leaky_relu(self.interest_transform_matrix(user_interest_t))

            transform_user_interest_t = F.leaky_relu(
                self.interest_transform_matrix_1(user_interest_t_1)) + F.leaky_relu(
                self.interest_transform_matrix_2(user_interest_t_2))

            transform_user_interest_t_expanded = torch.unsqueeze(transform_user_interest_t, dim=1)

            # 先不过LSTM看看效果怎么样
            if user_interest is None:
                user_interest = transform_user_interest_t_expanded
            else:
                user_interest = torch.concat([user_interest, transform_user_interest_t_expanded], dim=1)

        return user_interest

    def log2feats(self, user_interest):
        positions = torch.tile(torch.tensor(list(range(user_interest.shape[1]))), [user_interest.shape[0], 1]).to(
            self.device)
        # # 这个技巧在论文中没说明，我自己实现的代码中也没有，但是在源码中有
        user_interest *= self.hidden_units ** 0.5
        user_interest = user_interest + self.pos_embed(positions)
        user_interest = self.embed_dropout(user_interest)

        # casual mask
        # True代表这个位置被掩盖
        attn_mask = ~torch.tril(
            torch.ones(user_interest.shape[1], user_interest.shape[1], dtype=torch.bool, device=self.device))

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](user_interest)
            mha_outputs, _ = self.attention_layers[i](Q, user_interest, user_interest, attn_mask=attn_mask)

            user_interest = Q + mha_outputs

            user_interest = self.forward_layernorms[i](user_interest)
            user_interest = self.forward_layers[i](user_interest)

        log_feats = self.last_layernorm(user_interest)

        return log_feats

    def forward(self, user, seq, pos, neg, memory_h, memory_r, memory_t):

        total_h_embed, total_r_embed, total_t_embed = None, None, None

        # user_interest: (batch_size, maxlen ,hidden_units)
        # user_short_interest, updated_item_embed, h_embed_list, r_embed_list, t_embed_list = self.get_user_short_interest(
        #     seq, memory_h, memory_r, memory_t)

        user_short_interest, updated_item_embed = self.get_user_short_interest(seq, memory_h, memory_r, memory_t)

        # for hop in range(self.n_hop):
        #     h_embed_expanded = torch.unsqueeze(h_embed_list[hop], dim=2)
        #     r_embed_expanded = torch.unsqueeze(r_embed_list[hop], dim=2)
        #     t_embed_expanded = torch.unsqueeze(t_embed_list[hop], dim=2)
        #
        #     if total_h_embed is None:
        #         total_h_embed = h_embed_expanded
        #     else:
        #         total_h_embed = torch.concat([total_h_embed, h_embed_expanded], dim=2)
        #
        #     if total_r_embed is None:
        #         total_r_embed = r_embed_expanded
        #     else:
        #         total_r_embed = torch.concat([total_r_embed, r_embed_expanded], dim=2)
        #
        #     if total_t_embed is None:
        #         total_t_embed = t_embed_expanded
        #     else:
        #         total_t_embed = torch.concat([total_t_embed, t_embed_expanded], dim=2)

        user_interest = self.get_user_interest(user, user_short_interest)
        log_feats = self.log2feats(user_interest)

        # 直接不考虑短期和长期兴趣的结合(消融实验部分)
        # log_feats = self.log2feats(user_short_interest)

        pos_embed = self.entity_embed(pos)
        neg_embed = self.entity_embed(neg)

        pos_logits = torch.sum((log_feats * pos_embed), dim=-1)
        neg_logits = torch.sum((log_feats * neg_embed), dim=-1)

        return pos_logits, neg_logits
        # return pos_logits, neg_logits, total_h_embed, total_r_embed, total_t_embed

    def predict(self, user, seq, item_indices, memory_h, memory_r, memory_t):
        # user_short_interest, update_item_embed, _, _, _ = self.get_user_short_interest(seq, memory_h, memory_r,
        #                                                                                memory_t)
        user_short_interest, update_item_embed = self.get_user_short_interest(seq, memory_h, memory_r, memory_t)

        user_interest = self.get_user_interest(user, user_short_interest)

        log_feats = self.log2feats(user_interest)
        final_feats = log_feats[:, -1, :]

        item_embed = self.entity_embed(item_indices)

        logits = item_embed.matmul(final_feats.unsqueeze(-1)).squeeze(-1)

        return logits


class TransR(nn.Module):
    def __init__(self, entity_nums, relation_nums, embed_dim):
        super(TransR, self).__init__()

        self.entity_embed = nn.Embedding(entity_nums, embed_dim, padding_idx=0)
        self.relation_embed = nn.Embedding(relation_nums, embed_dim, padding_idx=0)
        # trans_M对应于RippleNet4SR里的relation_embed
        self.trans_M = nn.Embedding(relation_nums, embed_dim * embed_dim, padding_idx=0)

        self.embed_dim = embed_dim

        self._init_param()

    def _init_param(self):
        for name, param in self.named_parameters():
            try:
                nn.init.xavier_normal_(param.data)
            except:
                pass

    def forward(self, h, r, pos_t, neg_t):
        # (batch_size, embed_dim)
        h_embed = self.entity_embed(h)
        pos_t_embed, neg_t_embed = self.entity_embed(pos_t), self.entity_embed(neg_t)

        # (batch_size, embed_dim)
        r_embed = self.relation_embed(r)
        # (batch_size, embed_dim, embed_dim)
        W_r = self.trans_M(r).view(-1, self.embed_dim, self.embed_dim)

        trans_h = torch.squeeze(torch.bmm(torch.unsqueeze(h_embed, dim=1), W_r), dim=1)
        trans_pos_t = torch.squeeze(torch.bmm(torch.unsqueeze(pos_t_embed, dim=1), W_r), dim=1)
        trans_neg_t = torch.squeeze(torch.bmm(torch.unsqueeze(neg_t_embed, dim=1), W_r), dim=1)

        return trans_h, r_embed, trans_pos_t, trans_neg_t
