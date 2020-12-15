from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import  transforms
import pickle
import os
import os.path
import datetime
import numpy as np
from data.highDSSl_loader_cleanv1_4_4 import MyDataset,DataLoader
from utils.util import AverageMeter, accuracy
from tqdm import tqdm
import shutil
from models.resnet3d import ResNet, BasicBlock 
import matplotlib.pyplot as plt


def get_inplanes():
    return [64, 128, 256, 512]

def visualize_scenarios(data):
    for k in range(2):
        for i in range(3):
            a = data[k,:,i,:,:]
            aa = a.reshape(30,180)
            stdName = 'check_'+str(k)+'_' +str(i)+'.pdf'
            plt.imshow(aa)
            plt.savefig(stdName)

def train(epoch, model, device, dataloader, optimizer, exp_lr_scheduler, criterion, args):
    loss_record = AverageMeter()
    acc_record = AverageMeter()
    model.train()
    for batch_idx, (data, label) in enumerate(tqdm(dataloader(epoch))):
        # visualize_scenarios(data)
        data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True) # add this line
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
     
        # measure accuracy and record loss
        acc = accuracy(output, label)
        acc_record.update(acc[0].item(), data.size(0))
        loss_record.update(loss.item(), data.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, loss_record.avg, acc_record.avg))

    return loss_record

def test(model, device, dataloader,criterion, args):
    acc_record = AverageMeter()
    model.eval()
    total_loss = 0.0
    for batch_idx, (data, label) in enumerate(tqdm(dataloader())):
        #visualize_scenarios(data)
        data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True) # add this line
        output = model(data)
        loss = criterion(output, label)
        total_loss += loss.item()
 
        # measure accuracy and record loss
        acc = accuracy(output, label)
        acc_record.update(acc[0].item(), data.size(0))
    avg_loss = total_loss / len(dataloader())

    print('Test Acc: {:.4f}'.format(acc_record.avg))
    return acc_record,avg_loss 

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    # Training settings
    parser = argparse.ArgumentParser(description='Rot_resNet')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                                    help='disables CUDA training')
    parser.add_argument('--num_workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=1,
                                    help='random seed (default: 1)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--dataset_name', type=str, default='scenarios', help='options: highd, roundabout')
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/Scenarios')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--model_name', type=str, default='shufflenet_scenarios')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= os.path.join(args.exp_root, runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+'{}.pth'.format(args.model_name) 
    args.model_dir1 = model_dir+'/'+'{}.pth'.format('allepocj') 
    train_transforms = transforms.Compose([
        transforms.Resize((30, 180)),  # smaller edge to 128
        #transforms.RandomCrop(112),
        transforms.ToTensor()
    ])          
    custom_dataset_train = MyDataset([0,3,6,9], args.dataset_root,'train',trans=train_transforms)

    train_loader = DataLoader(dataset=custom_dataset_train,
       batch_size=args.batch_size,
       num_workers=args.num_workers,
       shuffle=True,trans=train_transforms)

    train_loader = DataLoader(dataset=custom_dataset_train, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers,trans=train_transforms)

    custom_dataset_test = MyDataset([0,3,6,9],  args.dataset_root,'test',trans=train_transforms)
    
    test_loader = DataLoader(dataset=custom_dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        shuffle=False,trans=train_transforms)

    model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_classes=24)
    model = model.to(device)
    print(model) 

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)

    criterion = nn.CrossEntropyLoss()

    best_acc = 0 
    for epoch in range(args.epochs +1):
        loss_record = train(epoch, model, device, train_loader, optimizer, exp_lr_scheduler, criterion, args)
        acc_record,avg_loss = test(model, device, test_loader,criterion, args)
        exp_lr_scheduler.step()
        
        is_best = acc_record.avg > best_acc 
        best_acc = max(acc_record.avg, best_acc)
        torch.save(model.state_dict(), args.model_dir1)

        if is_best:
            torch.save(model.state_dict(), args.model_dir)

if __name__ == '__main__':
    main()
