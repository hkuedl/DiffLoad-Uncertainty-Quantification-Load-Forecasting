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
from blitz.modules import BayesianLSTM,BayesianGRU,BayesianLinear
from blitz.utils import variational_estimator
import time

from scipy.stats import norm
from scipy.stats import cauchy
import properscoring as ps


class GRU(nn.Module):
    def __init__(self, input_size, n_hiddens, num_layers, output_size):
        super().__init__()
        self.n_input = input_size
        input_size = input_size
        self.num_layers = num_layers
        self.hiddens = n_hiddens
        self.n_output = output_size

        features = nn.ModuleList()
        for hidden in n_hiddens:
            rnn = nn.GRU(
                input_size=input_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
            )
            features.append(rnn)
            input_size = hidden
            self.features = nn.Sequential(*features)
        self.fc_out = nn.Linear(n_hiddens[-1],output_size)

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]
        seq_len = input_seq.shape[1]
        input_seq = input_seq.view(batch_size, seq_len, self.n_input)
        out = self.gru_features(input_seq)
        fea = out[0]
        output = self.fc_out(fea[:,-1,:])
        
        return output

    def gru_features(self, x, predict=False):
        x_input = x
        out = None
        out_lis = []
        for i in range(self.num_layers):
            out, hidden = self.features[i](x_input.float())
            x_input = out
            out_lis.append(out)
        return out, hidden, out_lis


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



class likelihood(nn.Module):
    def __init__(self):
        super(likelihood, self).__init__()
    def forward(self, label, mu, sigma):
        distribution = torch.distributions.normal.Normal(mu,sigma)
        loss = distribution.log_prob(label)

        return -torch.mean(loss)

def create_folder(root_path,t):
    if os.path.isdir(root_path)!=True:
        os.mkdir(root_path)
    if os.path.isdir(root_path+'/pkl_folder')!=True:
        os.mkdir(root_path+'/pkl_folder')
    path = root_path+'/pkl_folder/'+str(t)
    if os.path.isdir(path)!=True:
        os.mkdir(path)
    return(path)



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def MAPE(actual, pred): 
    actual, pred = np.array(actual.cpu().detach().numpy()), np.array(pred.cpu().detach().numpy())
    return np.mean(np.abs((actual - pred) / actual)) * 100
def MAE(true,value):
    diff = np.array(true-value)
    return(np.mean(np.abs(diff)))

def RMSE(true,value):
    return(torch.sqrt(torch.mean(torch.pow((true-value),2))))


def P2sigma(true,pred,sigma):
    upper = pred+sigma
    lower = pred-sigma
    index = np.mean((lower<=true)&(true<=upper))
    return(index)

def QCI(true,pred,sigma):
    error = np.abs(true-pred)
    inter = np.abs(sigma-error)
    return(np.mean(inter))


def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def CRPS(true,mu,sigma):
    distribution = torch.distributions.normal.Normal(torch.Tensor(mu),torch.Tensor(sigma))
    sample = distribution.sample([10000])
    return(np.mean(ps.crps_ensemble(true,sample.permute(1,0))))
    #return(np.mean(ps.crps_gaussian(true,mu,sigma)))

def eval_pinball(true,mu,sigma,quantiles):
    losses = []
    true = torch.Tensor(true)
    mu = torch.Tensor(mu)
    sigma = torch.Tensor(sigma)
    for k in range(len(true)):
        ppf_list = norm.ppf(q = quantiles,loc = mu[k],scale = sigma[k])
        for i, q in enumerate(quantiles):
            errors = true[k] - ppf_list[i]
            losses.append(torch.max((q - 1) * errors, q * errors))
    return(np.mean(losses))


def create_result(root_path,t):
    if os.path.isdir(root_path+'/result')!=True:
        os.mkdir(root_path+'/result')
    path = root_path+'/result/'+str(t)
    if os.path.isdir(path)!=True:
        os.mkdir(path)

    model_path = root_path+'/pkl_folder/'+str(t)+'/'+str(t)+'_.pkl'
    return(path,model_path)


def MAE(true,value):
    diff = np.array(true-value)
    return(np.mean(np.abs(diff)))
def RMSE(true,value):
    return(np.sqrt(np.mean(np.power((true-value),2))))

def P2sigma(true,mu,sigma,quantiles):
    seq = []
    for i in range(len(true)):
        metric = (true[i]<=norm.ppf(quantiles[1],mu[i],sigma[i]))&(true[i]>=norm.ppf(quantiles[0],mu[i],sigma[i]))
        seq.append(metric)
    return(np.mean(seq))

def QCI(true,mu,sigma):
    seq = []
    for i in range(len(true)):
        term1 = np.abs(true[i]-mu[i])
        term2 = norm.ppf(0.75,mu[i],sigma[i])-norm.ppf(0.25,mu[i],sigma[i])
        metric = np.abs(term1-term2)
        seq.append(metric)
    return(np.mean(seq))

def accuracy(pred,true,mu_std,interval = 24):
    length = len(pred)
    test_accuracy = []
    for i in np.arange(interval,length,interval):
        error1 = np.mean(np.abs(pred[i-interval:i]-true[i-interval:i])/np.abs(true[i-interval:i]))
        error2 = np.mean(np.abs(pred[i:i+interval]-true[i:i+interval])/np.abs(true[i:i+interval]))
        mu_std1 = np.mean(mu_std[i-interval:i]/pred[i-interval:i])
        mu_std2 = np.mean(mu_std[i:i+interval]/pred[i:i+interval])
        if (error1-error2)*(mu_std1-mu_std2)>0:
            test_accuracy.append(1)
        else:
            test_accuracy.append(0)
    return(np.mean(test_accuracy))


