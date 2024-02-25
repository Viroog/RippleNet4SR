import argparse
import os

import torch

from utils import load_data, WarpSampler, evaluate_valid, evaluate
from model import RippleNet4SR
import torch.optim as optim
from train_transR import train
import torch.nn.functional as F

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='books')
parser.add_argument('--maxlen', type=int, default=80)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--hidden_units', type=int, default=48)
parser.add_argument('--block_nums', type=int, default=2)
parser.add_argument('--head_nums', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.001)
# parser.add_argument('--kge_weight', type=float, default=0.01)
parser.add_argument('--l2', type=float, default=1e-4)
parser.add_argument('--item_update_mode', type=str, default='plus',
                    help='how to update item at the end of each hop')
parser.add_argument('--dropout_rate', type=float, default=0.3)
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--n_hop', type=int, default=3, help='the total hop of ripple net')
parser.add_argument('--use_ripple_padding', type=bool, default=False,
                    help='if there is no next ripple set, True uses padding else use last hop set')
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
parser.add_argument('--max_tolerant', type=int, default=3, help='if continue 4 performance not improved, stop it')
parser.add_argument('--use_pretrained', type=bool, default=False, help='whether to use pretrain TransR embedding')
parser.add_argument('--pretrained_model', type=str, default='pretrained_model.pth', help='the path of pretrained model')
parser.add_argument('--mode', type=str, default='train', help='train or just test')

args = parser.parse_args()

# 需要加这一句，否则多线程会报错
if __name__ == '__main__':
    pretrained_model = None
    # if args.use_pretrained:
    #     pretrained_model_path = f'./pretrained/{args.pretrained_model}'
    #     pretrained_model = torch.load(pretrained_model_path)
    #     pretrained_model.to(args.device)

    user_train, user_valid, user_test, user_nums, item_nums, entity_nums, relation_nums, ripple_set = load_data(args, pretrained_model)
    batch_num = len(user_train) // args.batch_size

    sampler = WarpSampler(user_train, ripple_set, user_nums, item_nums, args.batch_size, args.maxlen, args.n_hop,
                          args.n_memory, n_workers=3)

    pretrained_model = None
    if args.use_pretrained:
        pretrained_model_path = f'./pretrained/{args.pretrained_model}'
        # 如果存在则直接加载
        if os.path.exists(pretrained_model_path):
            pretrained_model = torch.load(pretrained_model_path)
        # 不存在重新开始训练
        else:
            train(entity_nums, relation_nums, pretrained_model_path)

    if args.mode == 'train':
        best_ndcg, best_hit = 0, 0
        tolerant = 0
        best_model = None

        model = RippleNet4SR(user_nums, entity_nums, relation_nums, pretrained_model, args).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        criterion = torch.nn.BCEWithLogitsLoss()

        model.train()

        for epoch in range(args.epochs):
            total_loss = 0
            for step in range(batch_num):
                # user:(128,)  seq==pos==neg:(128, 50)
                # seq_ripple_set:(128, 50, 2, 3, 32) ==> (batch_size, seq_len, n_hop, (head, relation, tail), n_memory)
                user, seq, pos, neg, seq_ripple_set = sampler.next_batch()
                user, seq, pos, neg, seq_ripple_set = np.array(user), np.array(seq), np.array(pos), np.array(neg), np.array(
                    seq_ripple_set)

                # memory_x: (128, 50, 2, 32)
                memory_h = seq_ripple_set[:, :, :, 0, :]
                memory_r = seq_ripple_set[:, :, :, 1, :]
                memory_t = seq_ripple_set[:, :, :, 2, :]

                user = torch.tensor(user, dtype=torch.int64).to(args.device)
                seq = torch.tensor(seq, dtype=torch.int64).to(args.device)
                pos = torch.tensor(pos, dtype=torch.int64).to(args.device)
                neg = torch.tensor(neg, dtype=torch.int64).to(args.device)
                memory_h = torch.tensor(memory_h, dtype=torch.int64).to(args.device)
                memory_t = torch.tensor(memory_t, dtype=torch.int64).to(args.device)
                memory_r = torch.tensor(memory_r, dtype=torch.int64).to(args.device)

                # pos_logits, neg_logits, total_h_embed, total_r_embed, total_t_embed = model(user, seq, pos,
                #                                                                             neg, memory_h,
                #                                                                             memory_r,
                #                                                                             memory_t)

                pos_logits, neg_logits, = model(user, seq, pos, neg, memory_h, memory_r, memory_t)

                # loss1: bce_loss
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                       device=args.device)
                indices = torch.where(pos != 0)

                # 换成bpr loss试试
                # loss = torch.sum(-F.logsigmoid(pos_logits[indices] - neg_logits[indices]), dim=0) / indices[0].shape[0]

                # 不计算padding的loss
                loss = criterion(pos_logits[indices], pos_labels[indices])
                loss += criterion(neg_logits[indices], neg_labels[indices])

                # # 不计算padding的loss
                # total_h_embed = total_h_embed[indices]
                # total_r_embed = total_r_embed[indices]
                # total_t_embed = total_t_embed[indices]
                #
                # # loss2: kge_loss
                # loss2 = 0
                # for hop in range(args.n_hop):
                #     h_expanded = torch.unsqueeze(total_h_embed[:, hop, :, :], dim=2)
                #     t_expanded = torch.unsqueeze(total_t_embed[:, hop, :, :], dim=3)
                #     hRt = torch.squeeze(
                #         torch.matmul(torch.matmul(h_expanded, total_r_embed[:, hop, :, :, :]), t_expanded))
                #     loss2 += torch.sigmoid(hRt).mean()
                # loss2 = -args.kge_weight * loss2
                #
                # # loss3: kge_l2_loss
                # loss3 = 0
                # for hop in range(args.n_hop):
                #     loss3 += (total_t_embed[:, hop, :, :] * total_h_embed[:, hop, :, :]).sum()
                #     loss3 += (total_r_embed[:, hop, :, :, :] * total_r_embed[:, hop, :, :, :]).sum()
                #     loss3 += (total_t_embed[:, hop, :, :] * total_t_embed[:, hop, :, :]).sum()
                # loss3 = args.l2 * loss3

                # loss = loss1 + loss2 + loss3

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"epoch: {epoch + 1}, total_loss={total_loss / batch_num}")

            # 20个epoch在验证集上验证指标
            if (epoch + 1) % 20 == 0:
                model.eval()
                valid_ndcg, valid_hit = evaluate_valid(model, user_train, user_valid, user_nums, item_nums, ripple_set,
                                                       args)
                test_ndcg, test_hit = evaluate(model, user_train, user_valid, user_test, user_nums, item_nums, ripple_set,
                                               args)
                print(f'validation: NDCG: {valid_ndcg}, HIT: {valid_hit}  test: NDCG: {test_ndcg}, HIT: {test_hit}')

                if (test_ndcg > best_ndcg) and (test_hit > best_hit):
                    best_ndcg, best_hit = test_ndcg, test_hit
                    tolerant = 0
                else:
                    tolerant += 1

                model.train()

            if tolerant >= args.max_tolerant:
                break

        print(f'best ndcg:{best_ndcg}, best hit: {best_hit}')
        torch.save(best_model,
                   f'./trained/{args.dataset}/{args.lr}_{args.maxlen}_{args.hidden_units}_{args.block_nums}_{args.head_nums}_{args.n_memory}.pth')

    elif args.mode == 'test':
        model = torch.load(f'./trained/{args.dataset}/{args.lr}_{args.maxlen}_{args.hidden_units}_{args.block_nums}_{args.head_nums}_{args.n_memory}.pth')

        # 测试代码