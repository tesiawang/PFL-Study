import random
import copy
import time
import math
import numpy as np
import torch
import torch.optim as optim
from fedavg.config import get_args
from model import simplecnn, textcnn
from test import compute_local_test_accuracy, compute_acc
from prepare_data import get_dataloader

def local_train(args, nets_this_round, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list):
    
    for net_id, net in nets_this_round.items():
        train_local_dl = train_local_dls[net_id]
        data_distribution = data_distributions[net_id]

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,
                                weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        net.cuda()
        net.train()
            
        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)

            x, target = x.cuda(), target.cuda()
            
            optimizer.zero_grad()
            target = target.long()

            out = net(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            
        val_acc = compute_acc(net, val_local_dls[net_id])
        
        personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)

        if val_acc > best_val_acc_list[net_id]:
            best_val_acc_list[net_id] = val_acc
            best_test_acc_list[net_id] = personalized_test_acc
        print('>> Client {} | Personalized Test Acc: {:.5f} | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
        net.to('cpu')
    return np.array(best_test_acc_list).mean()


args, cfg = get_args()
print(args)
seed = args.init_seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)

n_party_per_round = int(args.n_parties * args.sample_fraction)
party_list = [i for i in range(args.n_parties)]
party_list_rounds = []
if n_party_per_round != args.n_parties:
    for i in range(args.comm_round):
        party_list_rounds.append(random.sample(party_list, n_party_per_round))
else:
    for i in range(args.comm_round):
        party_list_rounds.append(party_list)

train_local_dls, val_local_dls, test_dl, net_dataidx_map, traindata_cls_counts, data_distributions = get_dataloader(args)

if args.dataset == 'cifar10':
    model = simplecnn
elif args.dataset == 'cifar100':
    model = simplecnn
elif args.dataset == 'yahoo_answers':
    model = textcnn
    
local_models = []
best_val_acc_list, best_test_acc_list = [],[]

for i in range(args.n_parties):
    local_models.append(model(cfg['classes_size']))
    best_val_acc_list.append(0)
    best_test_acc_list.append(0)
    

for round in range(args.comm_round):          # Federated round loop
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction<1.0:
        print(f'>> Clients in this round : {party_list_this_round}')

    nets_this_round = {k: local_models[k] for k in party_list_this_round}
    
    # Local Model Training
    mean_personalized_acc = local_train(args, nets_this_round, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list)

    print('>> (Current) Round {} | Local Per: {:.5f}'.format(round, mean_personalized_acc))
    print('-'*80)

 