import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import cluster_acc, Identity, AverageMeter
from tqdm import tqdm
import numpy as np
import os
import umap
import matplotlib.pyplot as plt
from models.resnet3d_finetune import ResNet, BasicBlock 
from data.highD_loader2_cleanv1_4_4 import MyDataset,ScenarioLoader
from torch import optim

def visualize_scenarios(data):
    for k in range(5):
        for i in range(3):
            a = data[k,:,i,:,:]
            aa = a.reshape(30,180)
            stdName = 'check_'+str(k)+'_' +str(i)+'.png'
            plt.imshow(aa)
            plt.savefig(stdName)  
def get_inplanes():
    return [64, 128, 256, 512]

def train(model, train_loader, labeled_eval_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss() 
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()

        represen_x = np.zeros((2400,512))
        train_y    = np.zeros((2400,))
        for batch_idx, (x,  label, idx) in enumerate(tqdm(train_loader)):            
            if batch_idx == 0:
                idx = idx.cpu().data.numpy()
                label = label.to(device)
                label = label.cpu().data.numpy()

                train_y[idx] = label
                x = x.to(device)
                o1,o2,represen = model(x)
                represen = represen.cpu().data.numpy()

                represen_x[idx,:] = represen
            else:
                idx = idx.cpu().data.numpy()
                label = label.to(device)
                label = label.cpu().data.numpy()

                train_y[idx] = label
                x = x.to(device)
                o1,o2,represen = model(x) 
                represen = represen.cpu().data.numpy()
                represen_x[idx,:] =  represen

        U = umap.UMAP(n_components = 2)
        print('Shape_Extracted', represen_x.shape)
        embedding2 = U.fit_transform(represen_x,)
        fig, ax = plt.subplots(1, figsize=(14, 10))

        plt.scatter(embedding2[:, 0], embedding2[:, 1], s= 5, c=train_y, cmap='Spectral')
        savename= 'Check_'+str(epoch)+'.png'
        plt.savefig(savename)
        for batch_idx, (x, label, idx) in enumerate(tqdm(train_loader)):
            #visualize_scenarios(x)
            x, label = x.to(device), label.to(device)
            output1, _, _ = model(x)
            loss= criterion1(output1, label)
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('test on labeled classes')
        args.head = 'head1'
        test(model, labeled_eval_loader, args)

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
    return preds 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--step_size', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_unlabeled_classes', default=3, type=int)
    parser.add_argument('--num_labeled_classes', default=4, type=int)
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/Scenarios')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--shufflenet_dir', type=str, default='./data/experiments/selfsupervised_learning_scenario/shufflenet_scenarios.pth')
    parser.add_argument('--model_name', type=str, default='shufflenet_scenarios')
    parser.add_argument('--dataset_name', type=str, default='scenarios', help='options: highd, roundabout')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
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

    state_dict = torch.load(args.shufflenet_dir)
    del state_dict['fc.weight']
    del state_dict['fc.bias']
    model.load_state_dict(state_dict, strict=False)
    for name, param in model.named_parameters(): 
        if 'head' not in name and 'layer4' not in name:
            param.requires_grad = False
        else:
            print(name)

    labeled_train_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', shuffle=True, aug='once', target_list = range(args.num_labeled_classes))
    labeled_eval_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', shuffle=False,aug=None, target_list = range(args.num_labeled_classes))
    
    if args.mode == 'train':
        train(model, labeled_train_loader, labeled_eval_loader, args)
        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))
    elif args.mode == 'test':
        print("model loaded from {}.".format(args.model_dir))
        model.load_state_dict(torch.load(args.model_dir))
    print('test on labeled classes')
    args.head = 'head1'
    test(model, labeled_eval_loader, args)
