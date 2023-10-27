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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)
import properscoring as ps
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy.stats import cauchy
import time

import sys
sys.path.append('.')
from utils import *


if __name__ == '__main__':
    root_path = './lr5'
    metric_list = [[],[],[],[],[],[]] 
    for t in range(1):
        t = t+1
        setup_seed(t)
        print("begin loading data")
        train_data_path = '../GEF_data/train_data.npy'
        train_label_path = '../GEF_data/train_label.npy'
        test_data_path = '../GEF_data/test_data.npy'
        test_label_path = '../GEF_data/test_label.npy'
        mytrain_data = np.load(train_data_path)
        cov = mytrain_data.shape[2]
        day = mytrain_data.shape[1]
        mytrain_data = mytrain_data.reshape(-1,cov)
        mytest_data = np.load(test_data_path)
        mytest_data = mytest_data.reshape(-1,cov)
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
        # new_test = mydataset(mytest_data,mytest_label)
        # test_loader = DataLoader(new_test, shuffle=False, batch_size=1)
        print("finish loading data")
        path,model_path = create_result(root_path,t)
        model = torch.load(model_path).to(device)

        
        my_test_data = torch.Tensor(mytest_data).to(device)
        my_test_label = torch.Tensor(mytest_label).reshape(-1,1).to(device)

        mean = sd_label.mean_
        scale = sd_label.scale_
        
        start_time = time.time()

        with torch.no_grad():
            mu, sigma = model(my_test_data)
        end_time = time.time() 


        run_time = end_time - start_time

        mu = mu.reshape(-1,1)
        sigma = sigma.reshape(-1,1)
        mytest_label = sd_label.inverse_transform(my_test_label.cpu().detach().numpy()).reshape(-1)
        mu = sd_label.inverse_transform((mu).cpu().detach().numpy()).reshape(-1)
        sigma = (sd_label.scale_*sigma.cpu().detach().numpy()).reshape(-1)

       

        test_pinball = eval_pinball(mytest_label,mu,sigma,[0.125,0.875])/0.25
        test_pinball2 = eval_pinball(mytest_label,mu,sigma,[0.25,0.75])/0.5
        test_pinball3 = eval_pinball(mytest_label,mu,sigma,[0.375,0.625])/0.75
        test_CRPS = CRPS(mytest_label,mu,sigma)
        test_MAE = MAE(mytest_label,mu)
        test_MAPE = MAPE(mytest_label,mu)
        print(t)
        print('CRPS',test_CRPS)
        print('MAPE',test_MAPE)
        print('MAE',test_MAE)
        print('pinball_loss75',test_pinball)
        print('pinball_loss50',test_pinball2)
        print('pinball_loss25',test_pinball3)
        # np.save(path+'/test_result.npy',sample)
        np.save(path+'/test_result_mean.npy',mu)
        np.save(path+'/test_label.npy',mytest_label)
        np.save(path+'/test_result_sigma.npy',sigma)
        metric_list[0].append(test_CRPS)
        metric_list[1].append(test_MAE)
        metric_list[2].append(test_MAPE)
        metric_list[3].append(test_pinball)
        metric_list[4].append(test_pinball2)
        metric_list[5].append(test_pinball3)
    print('DeepAR')
    result_dict = {}
    result_dict['GEF'] = [np.mean(metric_list[i]) for i in range(len(metric_list))]
    result_dict = pd.DataFrame(result_dict)
    result_dict.to_csv(root_path+'/result_deepAR_lr5.csv',index=False,sep = ',')
    
    
    


        