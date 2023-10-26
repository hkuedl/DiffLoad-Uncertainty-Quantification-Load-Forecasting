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
torch.cuda.set_device(2)
import properscoring as ps
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from scipy.stats import norm
from scipy.stats import cauchy
import time
import matplotlib.pyplot as plt
from scipy.stats import iqr
from utils import *






if __name__ == '__main__':
    diff_steps = 5
    root_path = './lr5_'+str(diff_steps)
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
        new_test = my_dataset(mytest_data,mytest_label)
        test_loader = DataLoader(new_test, shuffle=False, batch_size=1)
        print("finish loading data")


        num_steps = diff_steps
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

        path,model_path = create_result(root_path,t)
        model = torch.load(model_path).to(device)
        my_test_data = torch.Tensor(mytest_data).to(device)
        my_test_label = torch.Tensor(mytest_label).reshape(-1,1).to(device)
        mu_list = torch.zeros_like(my_test_label.reshape(-1))
        sigma_list = torch.zeros_like(my_test_label.reshape(-1))
        value_list = torch.zeros_like(my_test_label.reshape(-1))

        temp_list  = []
        start_time = time.time() 
        for i in range(100):
            with torch.no_grad():
                mu, sigma,_,value = model(my_test_data)
            mu_list = mu_list + mu
            sigma_list = sigma_list + sigma
            value_list = value_list + value
            temp_list.append(mu.cpu().detach().numpy())
        end_time = time.time() 
        run_time = end_time - start_time
        mu = mu_list/100
        sigma = sigma_list/100
        value = value_list/100

        # mu_std = np.std(np.array(temp_list),0)
        mu_std = iqr(np.array(temp_list), axis = 0,rng=(45,55), interpolation='midpoint')
        mu_std = sd_label.scale_*mu_std
        mu = mu.reshape(-1,1)
        sigma = sigma.reshape(-1,1)
        value = value.reshape(-1,1)
        error = sd_label.inverse_transform((my_test_label-value).cpu().detach().numpy()).reshape(-1)
        test_label = sd_label.inverse_transform(my_test_label.cpu().detach().numpy()).reshape(-1)
        result = sd_label.inverse_transform((value+mu).cpu().detach().numpy()).reshape(-1)
        value = sd_label.inverse_transform((value).cpu().detach().numpy()).reshape(-1)
        mu = sd_label.inverse_transform((mu).cpu().detach().numpy()).reshape(-1)
        sigma = (sd_label.scale_*sigma.cpu().detach().numpy()).reshape(-1)


        test_label = np.array(test_label,dtype=np.float64)
        mu = np.array(mu,dtype=np.float64)
        sigma = np.array(sigma,dtype=np.float64) + mu_std
        value = np.array(value,dtype=np.float64)
        result = np.array(result,dtype=np.float64)
        error = np.array(error,dtype=np.float64)
        test_pinball = eval_pinball(test_label,result,sigma,[0.125,0.875])/0.25
        test_pinball2 = eval_pinball(test_label,result,sigma,[0.25,0.75])/0.5
        test_pinball3 = eval_pinball(test_label,result,sigma,[0.375,0.625])/0.75
        test_CRPS = CRPS(test_label,result,sigma)
        test_MAE = MAE(test_label,result)
        test_MAPE = MAPE(test_label,result)

        print(t)
        print('CRPS',test_CRPS)
        print('MAPE',test_MAPE)
        print('MAE',test_MAE)
        print('pinball_loss75',test_pinball)
        print('pinball_loss50',test_pinball2)
        print('pinball_loss25',test_pinball3)
        # np.save(path+'/test_result.npy',sample)
        np.save(path+'/test_result_mean.npy',result)
        np.save(path+'/test_label.npy',test_label)
        np.save(path+'/test_result_sigma.npy',sigma)
        metric_list[0].append(test_CRPS)
        metric_list[1].append(test_MAE)
        metric_list[2].append(test_MAPE)
        metric_list[3].append(test_pinball)
        metric_list[4].append(test_pinball2)
        metric_list[5].append(test_pinball3)
    print('Diffusion')
    result_dict = {}
    result_dict['GEF'] = [np.mean(metric_list[i]) for i in range(len(metric_list))]
    # result_dict['GEF'].append(run_time)
    result_dict = pd.DataFrame(result_dict)
    result_dict.to_csv(path+'/result_seq2seq_diffusion_lr5.csv',index=False,sep = ',')
        
        


        