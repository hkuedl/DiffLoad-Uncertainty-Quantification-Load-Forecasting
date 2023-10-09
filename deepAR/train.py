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
import sys
sys.path.append('.')
from utils import *

    
if __name__ == '__main__':
    root_path = './lr5'
    for t in range(1):
        t = t+1
        setup_seed(t)
        batch_size = 256
        print("begin loading data")
        train_data_path = './GEF_data/train_data.npy'
        train_label_path = './GEF_data/train_label.npy'
        val_data_path = './GEF_data/val_data.npy'
        val_label_path = './GEF_data/val_label.npy'
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
        input_size, hidden_size, num_layers, output_size = 6, [64,64], 2, 1

        
        model = GRU(input_size, hidden_size, num_layers, output_size).to(device)
        loss_function = likelihood().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        epochs = 300

        count = 15
        val_min = np.inf

        for i in range(epochs):
            if count>15:
                break
            losses = []
            val_RMSE = []
            print('current_epoch', i)
            
            for (data, label) in train_loader:
                train_data = data.to(device)
                train_label = label.reshape(-1).to(device)
                
                

                mu, sigma = model(train_data)
                mu = mu.reshape(-1)
                sigma = sigma.reshape(-1)
                loss = loss_function(train_label,mu,sigma)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                losses.append(loss.item())

            for (data,label) in val_loader:
                val_data = data.to(device)
                val_label = label.reshape(-1).to(device)
                with torch.no_grad():
                    val_mu, val_sigma = model(val_data)
                val_mu = val_mu.reshape(-1)
                val_sigma = val_sigma.reshape(-1)
                val_loss = RMSE(val_label,val_mu)
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
                
        

        
            
            