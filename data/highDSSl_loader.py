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
from random import randint
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, stuff_in, mat_path, mode='train',trans = None):
        self.stuff = stuff_in
        self.transforms_ = trans
        if(mode=='train'):
            data1  = io.loadmat(mat_path+'\HighDScenarioClassV1_Train.mat')['X_train1']
            data2 = io.loadmat(mat_path+'\HighDScenarioClassV1_Train.mat')['X_train2'] 
            data  = np.concatenate((data1,data2))

            data_gt = io.loadmat(mat_path+'\HighDScenarioClassV1_Train.mat')['y_train'] 
            y_train = np.squeeze(np.array(data_gt))         
            self.images = data
            self.target = torch.from_numpy(y_train) 

        else:
            data = io.loadmat(mat_path+'\HighDScenarioClassV1_Test.mat')['X_test']
            data_gt = io.loadmat(mat_path+'\HighDScenarioClassV1_Test.mat')['y_test']   
            y_test = np.squeeze(np.array(data_gt))                
            self.images = data
            self.target = torch.from_numpy(y_test) 

    def get_shuffle_id(self,stuff):
        list_to_shuffle = []
        for L in range(0,len(stuff)+1):
            for subset in itertools.permutations(stuff,L):
                if(len(subset)>(len(stuff)-1)):
                    list_to_shuffle.append(subset)
        return list_to_shuffle             

    def generate_random_sequence(self,scenario, stuff):
        ssl_train = []
        list_to_shuffle = self.get_shuffle_id(stuff)
        trans_tuple = []

        for idx, list_check in enumerate(list_to_shuffle):
            trans_clip = []
            # fix seed, apply the sample `random transformation` for all frames in the clip 
            seed = random.random()
            for k in list_check:
                frame = scenario[:,:,k]*255.0
                random.seed(seed)
                frame = Image.fromarray(frame.astype('uint8'))
                frame = self.transforms_(frame) # tensor [C x H x W]
                trans_clip.append(frame)
            trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
            trans_tuple.append(trans_clip)
        tuple_clip = trans_tuple
        return tuple_clip 

    def correct_order_samples(self,stuff,train):
        ssl_train = []
        ssl_train.append(train[:,:,:,stuff])
        return ssl_train 
    def __getitem__(self, index):
        x = self.images[index]
        y = self.target[index]
        return x, y
 
    def __len__(self):
        return len(self.images)



def get_shuffle_id(stuff):
    list_to_shuffle = []
    for L in range(0,len(stuff)+1):
        for subset in itertools.permutations(stuff,L):
            if(len(subset)>(len(stuff)-1)):
                list_to_shuffle.append(subset)
    return list_to_shuffle             

def generate_random_sequence(scenario, stuff, tranform_):
    ssl_train = []
    list_to_shuffle = get_shuffle_id(stuff)
    trans_tuple = []

    for idx, list_check in enumerate(list_to_shuffle):
        trans_clip = []
        # fix seed, apply the sample `random transformation` for all frames in the clip 
        seed = random.random()
        for k in list_check:
            frame = scenario[:,:,k]*255.0
            random.seed(seed)
            frame = Image.fromarray(frame.astype('uint8'))
            frame = tranform_(frame) # tensor [C x H x W]
            trans_clip.append(frame)
        trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
        trans_tuple.append(trans_clip)
    tuple_clip = trans_tuple
    return tuple_clip 

class DataLoader(object):
    def __init__(self,
                  dataset,
                  batch_size=1,
                  epoch_size=None,
                  num_workers=0,
                  shuffle=True,
                  trans = None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = trans
    

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        def _load_function(idx):
            idx = idx % len(self.dataset)
            img0, _ = self.dataset[idx]
            x = [0,3,6,9]
            rotated_imgs = generate_random_sequence(img0, x, self.transform)
            rotation_labels = torch.LongTensor(list(range(24)))
            return torch.stack(rotated_imgs, dim=0), rotation_labels
        def _collate_fun(batch):
            batch = default_collate(batch)
            assert(len(batch)==2)
            batch_size, shuffle, channels, height, width, depth  = batch[0].size()
            batch[0] = batch[0].view([batch_size*shuffle, channels, height, width, depth])
            batch[1] = batch[1].view([batch_size*shuffle])
            return batch
      

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
            load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
            collate_fn=_collate_fun, num_workers=self.num_workers,
            shuffle=self.shuffle)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size
