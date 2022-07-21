from typing import List
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, get_worker_info
import numpy as np
import torch


class hdf5Dataset(Dataset):
    #dataset allowing patches from hdf5 file to be provided to pytorch dataloader.
    #also provides function to get a weighting over patches, based on h_channel,
    #or predictions of model for sampling MIL
    def __init__(self,hdf_path,transform=None, target_transform=None):
        
        self.transform = transform
        self.target_transform = target_transform
        self.means=[]
        self.stds=[]
        self.allpatches=[]
        self.needs_init=True
        self.hdf_path=hdf_path
        self.locs=[]
        self.f=[]
        self.ds=[]
        self.labels=[]
        self.weight_hist=[]  #for saving weight history if want to visualise evolution
        self.raw_hist=[]
        with h5py.File(self.hdf_path, 'r') as f:
            try:
                sizes=f['sizes']
                self.sizes=sizes[:]
            except:
                self.sizes=len(self.labels)

    def refresh_vds(self):
        self.needs_init=True
        self.f=[]
        self.ds=[]
        self.labels=[]
        self.sizes=[]
    
    def init_vds(self):
        self.needs_init=False
        self.f=h5py.File(self.hdf_path, 'r')
        self.ds=self.f['patches']
        try:
            self.labels=self.f['labels']
        except:
            print('no labels. Using dummmy label for feat extraction')
            sh=self.ds.shape
            self.labels=np.zeros((sh[0],))
        try:
            sizes=self.f['sizes']
            self.sizes=sizes[:]
        except:
            self.sizes=len(self.labels)

    def get_locs(self):
        if len(self.locs)==0:
            self.locs=self.f['locs'][:]
        return self.locs

    def close_hdf(self):
        if not isinstance(self.f, list):
            self.f.close()

    def __len__(self):
        with h5py.File(self.hdf_path, 'r') as f:
            l=len(f['patches'])
        return l

    def get_sizes(self):
        if len(self.sizes)==0:
            with h5py.File(self.hdf_path, 'r') as f:
                sizes=f['sizes'][:]
            return sizes
        else:
            return self.sizes

    def get_labels(self):
        if len(self.labels)==0:
            with h5py.File(self.hdf_path, 'r') as f:
                labels=f['labels'][:]
            return labels
        else:
            return self.labels
            
    def __getitem__(self, idx):
        if self.needs_init:
            self.init_vds()

        patch=self.ds[idx,:,:,:]
        label=self.labels[idx]
        view=False
        if view:
            plt.imshow(np.transpose(np.squeeze(patch),(1,2,0)))
        if False:
            self.allpatches.append(patch/255)
            self.means.append(np.mean(np.mean(patch/255,2),1))
            self.stds.append([np.std(patch[0,:,:]/255),np.std(patch[1,:,:]/255),np.std(patch[2,:,:]/255)])
        if self.transform:
            patch = self.transform(patch.copy())

        if self.target_transform:
            label = self.target_transform(label)

        return patch, label

    def get_weights(self,preds=None, alpha=2):
        
        with h5py.File(self.hdf_path, 'r') as f:
            sizes=f['sizes'][:]
            
            if preds==None:
                self.raw_hist.append(np.ones((np.sum(sizes),)))
            else:
                self.raw_hist.append(preds) #save weigths for vis
            slide_weights=[]
            ind=0
            for size in sizes:
                h=np.ones((size,))
                if preds!=None:
                    slide_preds=preds[ind:ind+size]**alpha
                    slide_preds=slide_preds+0.01
                    h=slide_preds.numpy()
                h=h*100/np.sum(h)  #normalises so all slides sampled evenly
                #h=h*size/np.sum(h)   #norm weights so select prob over slides prop to # patches in slide
                slide_weights.append(h)
            out=np.concatenate(slide_weights, axis=0)
            if True:
                self.weight_hist.append(out) #save weights for vis
            return out

    def get_mean_std(self):
        mean=np.mean(np.array(self.means), axis=0)
        std=np.mean(np.array(self.stds), axis=0)
        return mean, std

        


    

