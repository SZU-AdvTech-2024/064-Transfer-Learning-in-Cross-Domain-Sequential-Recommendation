import os
import time
import torch
import argparse
from tqdm import tqdm
from tjapl import TJAPL
from sasrec import SASRec
from utils import *


# print(torch.cuda.is_available())

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='sasrec')
parser.add_argument('--dataset', default='appliances',type=str)
parser.add_argument('--source1', default='electronics',type=str)
parser.add_argument('--source2', default='computers',type=str)
parser.add_argument('--train_dir', default='default',type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=100, type=int)
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.001, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--dataset_file',default='Rees46',type=str)

parser.add_argument('--is_only_overlap',default=True, type=str2bool)
parser.add_argument('--is_full_candidate',default=True, type=str2bool)
parser.add_argument('--is_test',default=True, type=str2bool)

args = parser.parse_args()
for arg_name, arg_value in vars(args).items():
    print(f"{arg_name}: {arg_value}")

if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'arg.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    dataset = data_partition_time(args.dataset, args.source1, args.source2,args.is_only_overlap ,args.is_full_candidate,args.dataset_file)  # datasets

    [user_train1, user_valid1, user_test1, usernum, itemnum1, user_neg1, user_train2, user_valid2, user_test2, itemnum2,
     user_train3, user_valid3, user_test3, itemnum3, time1, time2, time3,target_domain_all_items] = dataset
    num_batch = len(user_train1) // args.batch_size
    cc = 0.0
    for u in user_train1:
        cc += len(user_train1[u])
    print('average sequence length: %.2f' % (cc / len(user_train1)))
    print('usernum:',usernum,'itemnum1',itemnum1,'itemnum2',itemnum2,'itemnum3',itemnum3 )

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

    if args.is_test:
        user_train1_test= {}
        user_train2_test= {}
        user_train3_test= {}
        for u in range(1,usernum+1):
            user_train1_test[u] = user_train1[u] + user_valid1[u]
            user_train2_test[u] = user_train2[u] + user_valid2[u]
            user_train3_test[u] = user_train3[u] + user_valid3[u]

        sampler = WarpSampler(user_train1_test, user_train2_test, user_train3_test, time1, time2, time3, usernum, itemnum1,target_domain_all_items,
                          batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    else:
        sampler = WarpSampler(user_train1, user_train2, user_train3, time1, time2, time3, usernum,
                              itemnum1, target_domain_all_items,
                              batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    if args.model == 'sasrec':
        model=SASRec(usernum, itemnum1 + itemnum2 + itemnum3, args).to(args.device)
    elif args.model=='tjapl':
        model = TJAPL(usernum, itemnum1 + itemnum2 + itemnum3, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    model.train()

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb;

            pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    with tqdm(total=args.num_epochs * num_batch, desc=f"Training", leave=False, ncols=100) as pbar:
        #保存最好的指标
        best_ndcg={}
        best_hr={}
        best_k20=0
        best_epoch=0
        n=10
        for epoch in range(epoch_start_idx, args.num_epochs + 1):
            if args.inference_only: break  # just to decrease identition
            # for step in tqdm(range(num_batch), desc=f"Epoch {epoch}/{args.num_epochs}", leave=True, ncols=100):
            for step in range(num_batch):
                pbar.update(1)
                pbar.set_postfix({"Epoch": epoch})
                u, seq, pos, neg, seq2, mask2, seq3, mask3 = sampler.next_batch()
                u, seq, pos, neg, seq2, mask2, seq3, mask3 = np.array(u), np.array(seq), np.array(pos), np.array(
                    neg), np.array(seq2), np.array(mask2), np.array(seq3), np.array(mask3)
                if args.model=='sasrec':
                    pos_logits, neg_logits = model(u, seq,  pos, neg)
                elif args.model=='tjapl':
                    # pos_logits, neg_logits = model(u, seq, seq2, seq3, pos, neg, mask2, mask3)
                    pos_logits, neg_logits = model(u, seq, seq2, pos, neg, mask2)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                       device=args.device)

                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()

            if epoch % 20== 0 or epoch==1:
                model.eval()
                t1 = time.time() - t0
                T += t1
                print('\nepoch  %d :' % (epoch),end=' ')
                # print('Evaluating', end='')

                # t_test = evaluate(model, dataset, args)
                # t_valid = evaluate_valid(model, dataset, args)
                # print("Vaild:",end=' ')
                NDCG,Rec=evaluate_next(model,dataset,args,is_test=args.is_test,is_next_one=False)#验证集评估

                # 早停
                flag=False
                if NDCG[20]>=best_k20:
                    best_k20=NDCG[20]
                    flag=True
                # for k in NDCG.keys():
                #     if k not in best_ndcg:
                #         best_ndcg[k]=NDCG[k]
                #         best_hr[k]=Rec[k]
                #         flag=True
                #     if NDCG[k] > best_ndcg[k]:
                #         best_ndcg[k] = NDCG[k]
                #         flag=True
                #     if Rec[k] > best_hr[k]:
                #         best_hr[k] = Rec[k]
                #         flag=True
                if flag:
                    n=10
                    best_epoch=epoch
                    print("*")
                else:
                    n-=1
                    if n<=0:break

                # print('Test:', end=' ')
                # NDCG, Rec = evaluate_next(model, dataset, args, is_test=True,is_next_one=True)  # 测试集评估


                # print('\nepoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f) test (NDCG@10: %.4f, HR@10: %.4f)'
                #       % (epoch, T, t_valid[0], t_valid[1],
                #          t_test[0], t_test[1]))

                # f.write(str(t_valid) + ' ' + str(t_test) + '\n')
                # f.flush()
                t0 = time.time()
                model.train()
                # 将结果写入

            if epoch == args.num_epochs:
                folder = args.dataset + '_' + args.train_dir
                fname = 'TJAPL.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units,
                                     args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler.close()
    print("Done")
