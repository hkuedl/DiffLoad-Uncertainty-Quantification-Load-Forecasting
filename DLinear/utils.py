import os
import random
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import os
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from torch.nn import functional as F
import pickle
import random
from time import time
from typing import Union

import numpy as np
import torch
from torch import nn, optim
import torch.nn
from torch.nn.functional import mse_loss, l1_loss, binary_cross_entropy, cross_entropy
from torch.optim import Optimizer

class my_dataset(Dataset):
    def __init__(self, data ,label):

        self.seq = data
        self.label = label

        

    def __getitem__(self, idx):
        
        data_idx = torch.Tensor(deepcopy(self.seq[idx]))
        label_idx = torch.Tensor([deepcopy(self.label[idx])])
        
       
        
        return data_idx, label_idx

    def __len__(self):

        return len(self.seq)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#######Dlinear#######

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, seq_len,pred_len,enc_in = 1,enc_out = 1,individual=False):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size
        kernel_size = 7
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.out_channels = enc_out
        self.adjust_channels = nn.Linear(self.channels, self.out_channels)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = self.adjust_channels(x.permute(0,2,1))
        
        return x # to [Batch, Output length, Channel]
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def MAPE(actual, pred): 
    actual, pred = np.array(actual.cpu().detach().numpy()), np.array(pred.cpu().detach().numpy())
    return np.mean(np.abs((actual - pred) / actual)) * 100

def RMSE(true,value):
    return(torch.sqrt(torch.mean(torch.pow((true-value),2))))

def create_folder(root_path,t):
    if os.path.isdir(root_path)!=True:
        os.mkdir(root_path)
    if os.path.isdir(root_path+'/pkl_folder')!=True:
        os.mkdir(root_path+'/pkl_folder')
    path = root_path+'/pkl_folder/'+str(t)
    if os.path.isdir(path)!=True:
        os.mkdir(path)
    return(path)


def create_result(root_path,t):
    if os.path.isdir(root_path+'/result')!=True:
        os.mkdir(root_path+'/result')
    path = root_path+'/result/'+str(t)
    if os.path.isdir(path)!=True:
        os.mkdir(path)

    model_path = root_path+'/pkl_folder/'+str(t)+'/baseline_'+str(t)+'.pkl'
    return(path,model_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def MAPE(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100
def MAE(true,value):
    diff = np.array(true-value)
    return(np.mean(np.abs(diff)))

def RMSE(true,value):
    return(torch.sqrt(torch.mean(torch.pow((torch.Tensor([true])-torch.Tensor([value])),2))))