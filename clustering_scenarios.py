#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from utils.util import  PairEnum, cluster_acc, Identity, AverageMeter, seed_torch
from utils import ramps 
from tqdm import tqdm
from sklearn.ensemble import  RandomTreesEmbedding
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
from sklearn.cluster import KMeans
import gc 
import matplotlib
import tkinter
from models.resnet3d_finetune import ResNet, BasicBlock 
from data.highD_loader import ScenarioLoader, ScenarioLoaderMix
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def visualize_scenarios(data):
    for k in range(2):
        for i in range(3):
            a = data[k,:,i,:,:]
            aa = a.reshape(30,180)
            stdName = 'check_'+str(k)+'_' +str(i)+'.png'
            plt.imshow(aa)
            plt.savefig(stdName)

def get_inplanes():
    return [64, 128, 256, 512]

def train(model, train_loader, labeled_eval_loader, unlabeled_eval_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.BCELoss()
    for epoch in range(args.epochs): 
        loss_record = AverageMeter()
        model.train()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length) 
        represen_x = np.zeros((1600,512))
        train_y    = np.zeros((1600,))
        tr_x = np.zeros((2400,512))
        tr_y = np.zeros((2400,))
        for batch_idx, ((x, x_bar),  label, idx) in enumerate(tqdm(train_loader)):
            # visualize_scenarios(x)
            mask_rf = label<args.num_labeled_classes
            
            # Unknown
            Y_test = (label[~mask_rf]).detach()
            ulb_data = (x[~mask_rf]).detach()
            idx_ulb  = (idx[~mask_rf]).detach()
            idx_ulb  = idx_ulb.cpu().data.numpy()
            idx_ulb  = idx_ulb - 2400
            idx_ulb  = idx_ulb.astype(int) 
            
            # Known
            Y_train = (label[mask_rf]).detach()
            lb_data = (x[mask_rf]).detach()
            idx_lb  = (idx[mask_rf]).detach()
            idx_lb  = idx_lb.cpu().data.numpy()
            idx_lb  = idx_lb.astype(int)             

            # Unknown
            train_y[idx_ulb] = Y_test
            ulb_data = ulb_data.to(device)
            _,_,represen = model(ulb_data) 
            represen = represen.cpu().data.numpy()
            represen_x[idx_ulb,:] = represen
            
            # Known 
            tr_y[idx_lb] = Y_train
            lb_data = lb_data.to(device)
            _,_,rep_tr_known = model(lb_data) 
            rep_tr_known = rep_tr_known.cpu().data.numpy()
            tr_x[idx_lb,:] = rep_tr_known
            
           


        model_rf = RandomTreesEmbedding(n_estimators=500, n_jobs=-1,max_depth=None).fit(represen_x)
        print('Trees_Trained')
        model_rf.index(type_expect = 1)
        rfap = model_rf.encode_rfap(represen_x)    
        D = pairwise_distances(rfap, metric="hamming")
        S_epoch = 1 - D          
       
        kmeans = KMeans(n_clusters=3,n_init=20).fit(represen_x)         
        y = kmeans.labels_
        acc, nmi, ari = cluster_acc(train_y.astype(int), y.astype(int)), \
            nmi_score(train_y, y), ari_score(train_y, y) 
        print('K means acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))

            
        for batch_idx, ((x, x_bar),  label, idx) in enumerate(tqdm(train_loader)):         
            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)
            label = label.long()

            output1, output2, feat = model(x)
            output1_bar, output2_bar, _ = model(x_bar)
            prob1, prob1_bar, prob2, prob2_bar = F.softmax(output1, dim=1), F.softmax(output1_bar, dim=1), F.softmax(output2, dim=1), F.softmax(output2_bar, dim=1)
            
            mask_lb = label < args.num_labeled_classes

            rank_feat = (feat[~mask_lb]).detach()

            prob1_ulb, _ = PairEnum(prob2[~mask_lb])
            _, prob2_ulb = PairEnum(prob2_bar[~mask_lb])

            
            loss_ce = criterion1(output1[mask_lb], label[mask_lb])
            label[~mask_lb] = (output2[~mask_lb]).detach().max(1)[1] + args.num_labeled_classes
            loss_ce_add = w * criterion1(output1[~mask_lb], label[~mask_lb]) / args.rampup_coefficient *  args.increment_coefficient
        
            #Lakshman
            x1, x2  = PairEnum(rank_feat)
            x1 = x1.cpu().data.numpy()
            x2 = x2.cpu().data.numpy()
            idx = (idx[~mask_lb]).detach() 
            idx_ulb1 = idx.reshape(-1,1)

            idx1, idx2 = PairEnum(idx_ulb1)
            idx1 = idx1.cpu().data.numpy()
            idx2 = idx2.cpu().data.numpy()
            idx1 = idx1.astype(int)
            idx2 = idx2.astype(int)
            idx2 = idx2 - 2400          
            idx1 = idx1 - 2400


            similarity_drf = np.zeros((x1.shape[0],))
            for sima in (range(similarity_drf.shape[0])):
                similarity_drf[sima] = S_epoch[idx1[sima],idx2[sima]]
            similarity_drf = torch.from_numpy(similarity_drf).float().to(device)
                
            P = prob1_ulb.mul_(prob2_ulb)
            P = P.sum(1)            
            
            
            loss_bce = criterion2(P,similarity_drf)                       
            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)

            loss = loss_ce + loss_bce + loss_ce_add + w * consistency_loss

            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('test on labeled classes')
        args.head = 'head1'
        test(model, labeled_eval_loader, args)
        print('test on unlabeled classes')
        args.head='head2'
        acc = test(model, unlabeled_eval_loader, args)
        exp_lr_scheduler.step()
        print('Current learning rate is {}'.format(get_lr(optimizer)))


def test(model, test_loader, args):
    model.eval()
    preds=np.array([])
    targets=np.array([])
    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        output1, output2, _ = model(x)
        if args.head=='head1':
            output = output1
        else:
            output = output2
        _, pred = output.max(1)
        targets=np.append(targets, label.cpu().numpy())
        preds=np.append(preds, pred.cpu().numpy())
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds) 
    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    return acc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=225, type=int)
    parser.add_argument('--rampup_length', default=125, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=5)
    parser.add_argument('--increment_coefficient', type=float, default=0.05)
    parser.add_argument('--step_size', default=50 , type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_unlabeled_classes', default=3, type=int)
    parser.add_argument('--num_labeled_classes', default=4, type=int)
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/Scenarios')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--warmup_model_dir', type=str, default='./data/experiments/supervised_learning_scenarios/shufflenet_scenarios.pth')
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--IL', action='store_true', default=True, help='w/ incremental learning')
    parser.add_argument('--model_name', type=str, default='shufflenet_final')
    parser.add_argument('--dataset_name', type=str, default='scenarios', help='options: HighD, round')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()

 
  
    projectiondir = 'RF_trained'

    img_dir= os.path.join(args.exp_root, 'projection_trained_4_IL_final',projectiondir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    args.img_dir = img_dir
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= os.path.join(args.exp_root, runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+'{}.pth'.format(args.model_name) 
    model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(),
                num_labeled_classes = args.num_labeled_classes,
                num_unlabeled_classes = args.num_unlabeled_classes)
    model = model.to(device)
    num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    if args.mode=='train':
        state_dict = torch.load(args.warmup_model_dir)
        model.load_state_dict(state_dict)
        for name, param in model.named_parameters(): 
            if 'head' not in name and 'layer4' not in name:
                param.requires_grad = False
            else:
                print(name)


    mix_train_loader = ScenarioLoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice',shuffle=True, labeled_list=range(args.num_labeled_classes), unlabeled_list=range(args.num_labeled_classes, num_classes))
    labeled_train_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list = range(args.num_labeled_classes))
    unlabeled_eval_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
    unlabeled_eval_loader_test = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
    labeled_eval_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))
    all_eval_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(num_classes))

    if args.mode == 'train':
        save_weight = model.head1.weight.data.clone()
        save_bias = model.head1.bias.data.clone()
        model.head1 = nn.Linear(512, num_classes).to(device)
        model.head1.weight.data[:args.num_labeled_classes] = save_weight
        model.head1.bias.data[:] = torch.min(save_bias) - 1.
        model.head1.bias.data[:args.num_labeled_classes] = save_bias
        print(model)
        
        train(model, mix_train_loader, labeled_eval_loader, unlabeled_eval_loader, args)
        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))
    else:
        print("model loaded from {}.".format(args.model_dir))
        model.head1 = nn.Linear(512, num_classes).to(device)
        model.load_state_dict(torch.load(args.model_dir))

    print('Evaluating on Head1')
    args.head = 'head1'
    print('test on labeled classes (test split)')
    test(model, labeled_eval_loader, args)
    print('test on unlabeled classes (test split)')
    test(model, unlabeled_eval_loader_test, args)
    print('test on all classes (test split)')
    test(model, all_eval_loader, args)
    print('Evaluating on Head2')
    args.head = 'head2'
    print('test on unlabeled classes (train split)')
    acc_t = test(model, unlabeled_eval_loader, args)
    print('test on unlabeled classes (test split)')
    acc_t = test(model, unlabeled_eval_loader_test, args)
