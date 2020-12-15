from torch.utils.data.dataset import Dataset
import scipy.io as io
import torch
import tqdm
import torchvision.transforms as transforms
import numpy as np
import itertools
import random
from torch.utils.data.dataloader import default_collate
import torchnet as tnt
import torch.utils.data as data
from .utils import TransformTwice, GaussianBlur
from PIL import Image
import torchio

class MyDataset(Dataset):
    def __init__(self, stuff_in, mat_path, mode='train', target_list=range(4),
                 transform = None, aug=None):
        self.transform = transform
        self.transform_3d = transforms.Compose([torchio.transforms.RandomNoise(std=(0,0.00001))]) 

        self.stuff = stuff_in
        self.aaug = aug
        
        if(mode=='train'):
            data1  = io.loadmat(mat_path+'\HighDScenarioClassV1_Train.mat')['X_train1']
            data2 = io.loadmat(mat_path+'\HighDScenarioClassV1_Train.mat')['X_train2'] 
            data  = np.concatenate((data1,data2))
            data_gt = io.loadmat(mat_path+'\HighDScenarioClassV1_Train.mat')['y_train'] 
            y_train = np.squeeze(np.array(data_gt)) 
            
            X_train = self.correct_order_samples \
                (data)
            ind = [i for i in range(len(data_gt)) if data_gt[i] in target_list]
            train_set = X_train[ind]
            data_gt = data_gt[ind].tolist()
            data_gt = np.array(data_gt)
            data_gt = np.squeeze(data_gt)
            self.images = train_set
            self.target = torch.from_numpy(data_gt) 

        else:
            data = io.loadmat(mat_path+'\HighDScenarioClassV1_Test.mat')['X_test']
            data_gt = io.loadmat(mat_path+'\HighDScenarioClassV1_Test.mat')['y_test']   
            y_test = np.squeeze(np.array(data_gt))
            X_test = self.correct_order_samples \
                (data) 
            ind = [i for i in range(len(data_gt)) if data_gt[i] in target_list]
            test_set = X_test[ind]
            data_gt = data_gt[ind].tolist()
            data_gt = np.array(data_gt)
            data_gt = np.squeeze(data_gt)
            self.images = test_set                  
            self.target = torch.from_numpy(data_gt) 
            
      
              
    def correct_order_samples(self,train):
        ssl_train = []
        ssl_train = np.zeros((train.shape[0],30,200,4))
        for j in range(train.shape[0]):
            for k in range(4):
                ssl_train[j,:,:,k] = train[j,:,:,self.stuff[k]]
        return ssl_train 

    def __getitem__(self, index):
        x = self.images[index]
        y = self.target[index]
       # print('Data', x.shape)
        if self.aaug =='twice':
                # fix seed, apply the sample `random transformation` for all frames in the clip 
            seed = random.random()
            trans_clip1 = []
            trans_clip2 = []
            for k in range(4):
                frame = x[:,:,k]
                    #frame = frame.transpose()

                random.seed(seed)
                frame = Image.fromarray(frame.astype('uint8'))
                frame = self.transform(frame) # tensor [C x H x W]

                trans_clip1.append(frame)
                
            seed = random.random()            
            for k in range(4):
                frame = x[:,:,k]
                    #frame = frame.transpose()

                random.seed(seed)
                frame = Image.fromarray(frame.astype('uint8'))
                frame = self.transform(frame) # tensor [C x H x W]

                trans_clip2.append(frame)

                
            trans_clip1 = torch.stack(trans_clip1).permute([1, 0, 2, 3])
            trans_clip2 = torch.stack(trans_clip2).permute([1, 0, 2, 3])
            trans_clip1 = self.transform_3d(trans_clip1)
            trans_clip2 = self.transform_3d(trans_clip2)
   
            return (trans_clip1,trans_clip2), torch.tensor(int(y)), index  
                
        else:
            if self.transform:
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                trans_clip = []
                for k in range(4):
                    frame = x[:,:,k]
                   # print('Frame',frame.shape)    
                    random.seed(seed)
                    frame = Image.fromarray(frame.astype('uint8'))
                    frame = self.transform(frame) # tensor [C x H x W]
    
                    trans_clip.append(frame)
                    
                trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                if self.aaug is not None:
                    trans_clip = self.transform_3d(trans_clip)

            else:
                trans_clip = [torch.tensor(clip) for clip in x]
    
            return trans_clip, torch.tensor(int(y)), index  
 
    def __len__(self):
        return len(self.images)
    

    
def ScenarioLoaderMix(root, batch_size, split='train',num_workers=2, aug='None', shuffle=True, labeled_list=range(4), unlabeled_list=range(4, 7), new_labels=None):

    if aug==None:
        transform = transforms.Compose([
        transforms.Resize((30, 180)),  # smaller edge to 128
        transforms.ToTensor(),
        #binary_flip,
        #transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif aug=='once':
        transform = transforms.Compose([
        #transforms.RandomCrop((30,200)),
        #transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=(90, -90)),
        transforms.Resize((30, 180)),  # smaller edge to 128
        transforms.RandomCrop((30,180)),
        transforms.ToTensor(),
        #transforms.RandomErasing(),
        ])
    elif aug=='twice':
        transform = transforms.Compose([
        #transforms.RandomCrop((30,200)),
        #transforms.RandomHorizontalFlip(),
        transforms.Resize((30, 180)),  # smaller edge to 128
        transforms.RandomCrop((30,180)),
        transforms.ToTensor(),
        #transforms.RandomErasing(),
        ])   

    dataset_labeled = MyDataset(stuff_in=[0,3,6,9],mat_path=root, transform=transform, mode=split, target_list=labeled_list, aug=aug)
    dataset_unlabeled = MyDataset(stuff_in=[0,3,6,9],mat_path=root, transform=transform, mode=split, target_list=unlabeled_list, aug=aug)
    dataset_labeled.target = np.concatenate((dataset_labeled.target,dataset_unlabeled.target))
    dataset_labeled.images = np.concatenate((dataset_labeled.images,dataset_unlabeled.images),0)
    loader = data.DataLoader(dataset_labeled, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print('Data shape', dataset_labeled.images.shape)
    
    return loader

def ScenarioLoader(root, batch_size, split='train',aug='None', num_workers=2, shuffle=True, target_list=range(4)):

    if aug==None:
        transform = transforms.Compose([
        transforms.Resize((30, 180)),  # smaller edge to 128
        transforms.ToTensor(),
        #binary_flip,
        #transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif aug=='once':
        transform = transforms.Compose([
        #transforms.RandomCrop((30,200)),
        #transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=(90, -90)),
        transforms.Resize((30, 180)),  # smaller edge to 128
        transforms.RandomCrop((30,180)),
        transforms.ToTensor(),
        #transforms.RandomErasing(),
        ])
    elif aug=='twice':
        transform = transforms.Compose([
        #transforms.RandomCrop((30,200)),
        #transforms.RandomHorizontalFlip(),
        transforms.Resize((30, 180)),  # smaller edge to 128
        transforms.RandomCrop((30,180)),
        transforms.ToTensor(),
        #transforms.RandomErasing(),
        ])

    dataset = MyDataset(stuff_in=[0,3,6,9],mat_path=root, transform=transform, mode=split, target_list=target_list, aug=aug)
    loader = data.DataLoader(dataset, batch_size=batch_size,  shuffle=shuffle, num_workers=num_workers)
    print('Data shape', dataset.images.shape)

    return loader