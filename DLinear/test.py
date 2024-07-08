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
torch.cuda.set_device(2)
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
from utils import *



if __name__ == '__main__':
    root_path = './lr5'
    metric_list = [[],[],[]] 
    for t in range(1):
        t = t+1
        setup_seed(t)
        print("begin loading data")
        train_data_path = '../GEF_data/train_data.npy'
        train_label_path = '../GEF_data/train_label.npy'
        test_data_path = '../GEF_data/val_data.npy'
        test_label_path = '../GEF_data/val_label.npy'
        mytrain_data = np.load(train_data_path)
        cov = mytrain_data.shape[2]
        day = mytrain_data.shape[1]
        mytrain_data = mytrain_data.reshape(-1,1)
        mytest_data = np.load(test_data_path)
        mytest_data = mytest_data.reshape(-1,1)
        mytrain_label = np.load(train_label_path).reshape(-1,1)
        mytest_label = np.load(test_label_path).reshape(-1,1)
        sd_data = StandardScaler().fit(mytrain_data)
        sd_label = StandardScaler().fit(mytrain_label)
        mytrain_data = sd_data.transform(mytrain_data)
        mytest_data = sd_data.transform(mytest_data)
        mytrain_label = sd_label.transform(mytrain_label)
        mytest_label = sd_label.transform(mytest_label)
        mytrain_data = mytrain_data.reshape(-1,day,cov)
        mytest_data = mytest_data.reshape(-1,day,cov)
        mytrain_label = mytrain_label.reshape(-1)
        mytest_label = mytest_label.reshape(-1)
        print("finish loading data")
        path,model_path = create_result(root_path,t)
        model = torch.load(model_path).to(device)
        test_data = torch.Tensor(mytest_data).to(device)
        test_label = torch.Tensor(mytest_label).reshape(-1,1).to(device)
        with torch.no_grad():
            y_pred = model(test_data)
        y_pred = y_pred.reshape(-1,1)
        test_label = sd_label.inverse_transform(test_label.cpu().detach().numpy()).reshape(-1)
        y_pred = sd_label.inverse_transform(y_pred.cpu().detach().numpy()).reshape(-1)
        test_label = np.array(test_label,dtype=np.float64)
        y_pred = np.array(y_pred,dtype=np.float64)
        test_MAE = MAE(test_label,y_pred)
        test_MAPE = MAPE(test_label,y_pred)
        print(t)
        print('MAPE',test_MAPE)
        print('MAE',test_MAE)
        true = pd.DataFrame(test_label)
        value = pd.DataFrame(y_pred)

        metric_list[0].append(test_MAE)
        metric_list[1].append(test_MAPE)
        
        true.to_csv(path+'/'+str(t)+'_true.csv',index=False)
        value.to_csv(path+'/'+str(t)+'_value.csv',index=False)
        print('nbeats')
        result_dict = {}     
        print('test_MAE:',np.mean(metric_list[0]))
        print('test_MAPE:',np.mean(metric_list[1]))
        result_dict['GEF'] = [np.mean(metric_list[0]),np.mean(metric_list[1])]
        result_dict = pd.DataFrame(result_dict)
        result_dict.to_csv(root_path+'/result_DLinear.csv',index=False,sep = ',')
 
