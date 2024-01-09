import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__old(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)
        
    def __getitem__(self, idx): 
        with h5py.File(self.h5_file, 'r') as f: 
            lr = np.expand_dims(f['lr'][idx] / 255., 0) 
            hr = np.expand_dims(f['hr'][idx] / 255., 0) 
            hr_seg = np.expand_dims(f['segmentation'][idx], 0) 
            #hr_opt_flow = np.expand_dims(f['optical_flow'][idx], 0) 
            hr_depth = np.expand_dims(f['depth'][idx], 0) 
             
            return (lr, hr_seg, hr_depth), hr 

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
