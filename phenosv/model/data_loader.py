import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np
import pandas as pd
import torch.distributed as dist
from torch.utils.data import DistributedSampler, Dataset, Sampler
import math
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader



def pad_array(input_array, length):
    l = length - input_array.shape[0]
    m = nn.ZeroPad2d((0, 0, 0, l))
    return m(input_array)
def pad_list(input_list, length):
    l = length - input_list.shape[0]
    out = torch.cat((input_list, torch.zeros(l)), dim=0)
    return out

## for main fn model
def collate_fn_padd_local(batch):
    t1, t2, t3, t4 = zip(*batch)#x, gene_indicator, sv_indicator, y

    length_max = np.max([t.shape[0] for t in t1])

    mask = [torch.ones(t.shape) for t in t1]
    x = [pad_array(torch.tensor(t), length_max) for t in t1]
    gene_indicators = [pad_list(torch.tensor(t), length_max) for t in t2]
    sv_indicators = [pad_list(torch.tensor(t), length_max) for t in t3]
    y=list(t4)

    x = torch.stack(x)
    gene_indicators = torch.stack(gene_indicators)
    sv_indicators = torch.stack(sv_indicators)
    y = torch.tensor(np.array(y))

    ## compute mask
    mask = [pad_array(m, length_max) for m in mask]
    mask = torch.stack(mask)

    x = x.float()
    mask = mask.float()

    gene_indicators = gene_indicators.float()
    sv_indicators = sv_indicators.float()
    y = y.float()
    return x, mask, gene_indicators, sv_indicators, y

#weighted sampler
class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, sample_weights, total_samples, shuffle_data = True, num_replicas=None, rank=None, replacement=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.shuffle = shuffle_data
        self.sample_weights=torch.as_tensor(sample_weights, dtype=torch.double)
        self.total_samples=total_samples#total number for weighted sampling
        self.dataset = dataset
        self.num_replicas = num_replicas#gpu number
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.num_sample_each = int(math.ceil(self.total_samples*1.0 / self.num_replicas))

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples# sub dataset sample number

        # weighted sampler
        rand_tensor = torch.multinomial(self.sample_weights[indices], self.num_sample_each, self.replacement).tolist()

        return iter(rand_tensor)

    def __len__(self):
        return self.num_sample_each

    def set_epoch(self, epoch):
        self.epoch = epoch

##main model
class SVLocalDataset(data_utils.Dataset):
    #dataframe requires columns of
    # 1. label: 'PATHO'
    # 2. path of sv features: 'PATH',
    # 3. path_pheno, colname of storing of pheno_gene scores set to None if no in path_pheno;
    # 4. path_genescore: path to add sv-gene pair scores

    def __init__(self, dataframe, y_col='PATHO', perm = None,  feature_subset=None, ref_mat_path=None):
        super().__init__()
        self.df = dataframe
        self.y_col = y_col
        self.feature_subset = feature_subset
        self.perm = perm
        self.ref_mat_path = ref_mat_path
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        obj = np.load(self.df.loc[index, 'PATH'])
        x = np.squeeze(obj['feature_array'])
        if len(x.shape)==1:
            x = np.expand_dims(x, 0)

        gene_indicators = obj['gene_indicator']
        y = np.array(self.df.loc[index, self.y_col])
        element_names = obj['element_names']
        sv_indicator = [1 if 'sv' in e else 0 for e in element_names]
        if np.sum(sv_indicator)==0:
            sv_indicator = [1]*len(element_names)
        if self.perm is not None:
            mat = np.load(self.ref_mat_path)
            x[..., int(self.perm)] = np.random.choice(mat[...,int(self.perm)],x.shape[:-1])
        if self.feature_subset is not None:
            x = x[...,self.feature_subset]
        return x, gene_indicators, sv_indicator, y

class DataModule(pl.LightningDataModule):
    def __init__(self, train_df=None, vali_df=None, test_df=None,
                 batch_size=32, train_sampler_col = None, vali_sampler_col=None,
                 num_samples=10000, vali_samples = None, loader_workers=0, distributed = True,
                 y_col='PATHO', perm = None, feature_subset=None, ref_mat_path=None):
        super().__init__()

        self.train_sampler_col = train_sampler_col
        self.vali_sampler_col = vali_sampler_col

        if self.train_sampler_col is not None:
            self.train_df = train_df[train_df[train_sampler_col]>0].reset_index(drop=True)
        else:
            self.train_df = train_df

        if self.vali_sampler_col is not None:
            self.vali_df = vali_df[vali_df[vali_sampler_col]>0].reset_index(drop=True)
        else:
            self.vali_df = vali_df

        self.test_df = test_df
        self.num_samples = num_samples
        self.vali_samples=vali_samples
        self.loader_workers = loader_workers
        self.distributed = distributed
        self.batch_size = batch_size
        self.type=type
        self.y_col=y_col
        self.feature_subset=feature_subset
        self.perm = perm
        self.ref_mat_path=ref_mat_path

    def setup(self, stage=None):
        self.collate_function = collate_fn_padd_local

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.dset_train = SVLocalDataset(self.train_df, y_col=self.y_col, feature_subset=self.feature_subset)
            self.dset_vali = SVLocalDataset(self.vali_df, y_col=self.y_col, feature_subset=self.feature_subset)


        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.dset_test = SVLocalDataset(self.test_df, y_col=self.y_col, feature_subset=self.feature_subset)


        if stage == "predict" or stage is None:
            self.dset_test = SVLocalDataset(self.test_df, y_col=self.y_col, feature_subset=self.feature_subset,
                                            perm=self.perm, ref_mat_path=self.ref_mat_path)

    def train_dataloader(self):
        if self.train_sampler_col is not None:
            sample_weights = torch.tensor(np.array(self.train_df[self.train_sampler_col]))
            if self.distributed:
                sampler = DistributedWeightedSampler(self.dset_train, sample_weights, self.num_samples,
                                                     num_replicas=None, rank=None, replacement=True)
            else:
                sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=self.num_samples,
                                                                 replacement=True)
            dat_loader = data_utils.DataLoader(self.dset_train, batch_size=self.batch_size, pin_memory=True, sampler=sampler,
                                                   num_workers=self.loader_workers, collate_fn=self.collate_function)
        else:
            dat_loader = data_utils.DataLoader(self.dset_train,  batch_size=self.batch_size, shuffle=True, pin_memory=True,
                                               num_workers=self.loader_workers,collate_fn=self.collate_function)
        return dat_loader

    def val_dataloader(self):
        if self.vali_sampler_col is not None:
            sample_weights = torch.tensor(np.array(self.vali_df[self.vali_sampler_col]))
            if self.vali_samples is None:
                sample_number = int(2*np.sum(self.vali_df[self.y_col] == 1))
            else:
                sample_number = self.vali_samples
            if self.distributed:
                sampler = DistributedWeightedSampler(self.dset_vali, sample_weights, sample_number,
                                                     num_replicas=None, rank=None, replacement=True)
            else:
                sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=sample_number,replacement=True)

            dat_loader = data_utils.DataLoader(self.dset_vali, batch_size=self.batch_size, pin_memory=True, sampler=sampler,
                                                   num_workers=self.loader_workers, collate_fn=self.collate_function)
        else:
            dat_loader = data_utils.DataLoader(self.dset_vali, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                                           num_workers=self.loader_workers, collate_fn=self.collate_function)
        return dat_loader

    def test_dataloader(self):
        dat_loader = data_utils.DataLoader(self.dset_test, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                                           num_workers=self.loader_workers, collate_fn=self.collate_function)

        return dat_loader

    def predict_dataloader(self):
        dat_loader = data_utils.DataLoader(self.dset_test, batch_size=1, shuffle=False, collate_fn=self.collate_function)
        return dat_loader



