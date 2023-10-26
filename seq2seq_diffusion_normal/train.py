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
torch.cuda.set_device(3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch.distributions import normal,cauchy, kl_divergence

import time
from utils import *

if __name__ == '__main__':
    diff_steps = 5
    root_path = './lr5_'+str(diff_steps)
    for t in range(1):
        t = t+1
        setup_seed(t)
        batch_size = 256
        print("begin loading data")
        train_data_path = '../GEF_data/train_data.npy'
        train_label_path = '../GEF_data/train_label.npy'
        val_data_path = '../GEF_data/val_data.npy'
        val_label_path = '../GEF_data/val_label.npy'
        mytrain_data = np.load(train_data_path)
        cov = mytrain_data.shape[2]
        day = mytrain_data.shape[1]
        mytrain_data = mytrain_data.reshape(-1,cov)
        myval_data = np.load(val_data_path)
        myval_data = myval_data.reshape(-1,cov)
        mytrain_label = np.load(train_label_path).reshape(-1,1)
        myval_label = np.load(val_label_path).reshape(-1,1)
        sd_data = StandardScaler().fit(mytrain_data)
        sd_label = StandardScaler().fit(mytrain_label)
        mytrain_data = sd_data.transform(mytrain_data)
        myval_data = sd_data.transform(myval_data)
        mytrain_label = sd_label.transform(mytrain_label)
        myval_label = sd_label.transform(myval_label)
        mytrain_data = mytrain_data.reshape(-1,day,cov)
        myval_data = myval_data.reshape(-1,day,cov)
        mytrain_label = mytrain_label.reshape(-1)
        myval_label = myval_label.reshape(-1)
        new_train = my_dataset(mytrain_data,mytrain_label)
        new_val = my_dataset(myval_data,myval_label)
        train_loader = DataLoader(new_train, shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(new_val, shuffle=True, batch_size=batch_size)
        print("finish loading data")
        path  = create_folder(root_path,t)
        input_size, hidden_size, num_layers, output_size, num_steps = 6, [64,64], 2, 1,diff_steps

        #制定每一步的beta
        betas = torch.linspace(-6,6,num_steps).to(device)
        betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5

        #计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
        alphas = 1-betas
        alphas_prod = torch.cumprod(alphas,0).to(device)
        alphas_prod_p = torch.cat([torch.tensor([1]).float().to(device),alphas_prod[:-1].to(device)],0).to(device)
        alphas_bar_sqrt = torch.sqrt(alphas_prod).to(device)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod).to(device)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod).to(device)
        

        assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
        alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
        ==one_minus_alphas_bar_sqrt.shape
        print("all the same shape",betas.shape)
        
        model = GRU(input_size, hidden_size, num_layers, output_size,num_steps).to(device)
        loss_function = likelihood().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        epochs = 300

        count = 15
        val_min = np.inf
        model.train()
        start_time = time.time()
        for i in range(epochs):
            if count>15:
                break
            losses = []
            val_RMSE = []
            print('current_epoch', i)
            
            for (data, label) in train_loader:
                train_data = data.to(device)
                train_label = label.reshape(-1).to(device)
                
                

                mu, sigma,diff_loss,main_value = model(train_data)
                mu = mu.reshape(-1)
                sigma = sigma.reshape(-1)

                loss = loss_function(train_label-main_value,mu,sigma,diff_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                losses.append(loss.item())

            for (data,label) in val_loader:
                val_data = data.to(device)
                val_label = label.reshape(-1).to(device)
                with torch.no_grad():
                    val_mu, val_sigma,_,val_value = model(val_data)
                val_mu = val_mu.reshape(-1)
                val_sigma = val_sigma.reshape(-1)
                val_value = val_value.reshape(-1)
                val_loss = RMSE(val_label,val_value+val_mu)
                val_RMSE.append(val_loss.item())

            loss_av = np.mean(losses)
            val_av = np.mean(val_RMSE)
            print('train_loss:',loss_av)
            print('val_RMSE:',val_av)
            
            
            if (val_av<val_min):
                val_min = val_av
                count = 0
                torch.save(model, path+'/'+str(t)+'_.pkl')
            else:
                count = count+1
                print('count:',count)
        
        end_time = time.time()
        run_time = end_time - start_time
        
                
        

        
            
            