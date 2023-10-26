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
num_steps = 5

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

#计算任意时刻的x采样值，基于x_0和重参数化
def q_x(x_0,t):
    """可以基于x[0]得到任意时刻t的x[t]"""
    #noise = torch.randn_like(x_0)
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise)#在x[0]的基础上添加噪声

def p_sample_loop(model,shape,n_steps,betas,one_minus_alphas_bar_sqrt):
    """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model,cur_x,i,betas,one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq

def p_sample(model,x,t,betas,one_minus_alphas_bar_sqrt):
    """从x[T]采样t时刻的重构值"""
    x = x.to(device)

    coeff = betas[t]/ one_minus_alphas_bar_sqrt[t]

    t = torch.tensor([t]).to(device)

    
    eps_theta = model(x,t)
    
    mean = (1/(1-betas[t]).sqrt())*(x-(coeff*eps_theta))
    
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    
    sample = mean + sigma_t * z
    
    return (sample)

def diffusion_loss_fn(model,x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps):
    """对任意时刻t进行采样计算loss"""
    batch_size = x_0.shape[0]

    #对一个batchsize样本生成随机的时刻t
    t = torch.randint(0,n_steps,size=(batch_size//2,))
    t = torch.cat([t,n_steps-1-t],dim=0)
    t = t.unsqueeze(-1)

    #x0的系数
    a = alphas_bar_sqrt[t]

    #eps的系数
    aml = one_minus_alphas_bar_sqrt[t]

    #生成随机噪音eps
    e = torch.randn_like(x_0)

    #构造模型的输入
    x = x_0*a+e*aml

    #送入模型，得到t时刻的随机噪声预测值
    output = model(x,t.squeeze(-1))

    #与真实噪声一起计算误差，求平均值
    return (e - output).square().mean()

class MLPDiffusion(nn.Module):
    def __init__(self,n_steps,hidden_state,num_units=128):
        super(MLPDiffusion,self).__init__()
        
        self.linears = nn.ModuleList(
            [
                nn.Linear(hidden_state,num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.ReLU(),
                nn.Linear(num_units,hidden_state),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
            ]
        )
    def forward(self,x,t):
#         x = x_0
        for idx,embedding_layer in enumerate(self.step_embeddings):
        #t是代表加噪的步骤
            t_embedding = embedding_layer(t.to(device))
            x = self.linears[2*idx](x.to(device))
            x += t_embedding
            x = self.linears[2*idx+1](x)
            
        x = self.linears[-1](x)
        
        return x




class GRU(nn.Module):
    def __init__(self, input_size, n_hiddens, num_layers, output_size,num_steps):
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
        decoder = nn.ModuleList()
        input_size = 6
        for hidden in n_hiddens:
            rnn = nn.GRU(
                input_size=input_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
            )
            decoder.append(rnn)
            input_size = hidden
            self.decoder = nn.Sequential(*decoder)
        self.reverse = MLPDiffusion(num_steps,n_hiddens[-1])

        self.main = nn.Linear(n_hiddens[-1],output_size)
        self.distribution_presigma = nn.Linear(n_hiddens[-1], output_size)
        self.distribution_mu = nn.Linear(n_hiddens[-1], output_size)
        self.distribution_sigma = nn.Softplus()
        self.num_steps = num_steps

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]
        seq_len = input_seq.shape[1]
        input_seq = input_seq.view(batch_size, seq_len, self.n_input)
        _,hidden,_ = self.gru_features(input_seq,self.features)
        hidden_bfencode = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        hidden_bfencode = q_x(hidden_bfencode,self.num_steps-1)
        diff_loss = diffusion_loss_fn(self.reverse,hidden_bfencode,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,self.num_steps-1)
        
        hidden_afencode = p_sample(self.reverse,hidden_bfencode,self.num_steps-1,betas,one_minus_alphas_bar_sqrt)
        

        hidden_afencode = hidden_afencode.reshape(1,hidden_afencode.shape[0],-1)
        out = self.decoders(input_seq,hidden_afencode,self.features)
        fea = out[0]
        hidden = out[1]
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        main_value = self.main(hidden_permute)
        main_value = main_value.unsqueeze(1)
        main_value = main_value.expand(main_value.shape[0],7,1)
        new_input_seq = deepcopy(input_seq)
        new_input_seq[:,:,1] = new_input_seq[:,:,1]-main_value[:,:,0]
        _,hidden,_ = self.gru_features(new_input_seq,self.decoder)
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        pre_sigma = self.distribution_presigma(hidden_permute)
        mu = self.distribution_mu(hidden_permute)
        sigma = self.distribution_sigma(pre_sigma)
        mu = torch.squeeze(mu)
        sigma = torch.squeeze(sigma)
        return mu,sigma,diff_loss,main_value[:,0,0]

    def gru_features(self, x, model,predict=False):
        x_input = x
        out = None
        out_lis = []
        for i in range(self.num_layers):
            out, hidden = model[i](x_input.float())
            x_input = out
            out_lis.append(out)
        return out, hidden, out_lis

    
    def decoders(self, x,hidden,model,predict=False):
        x_input = x
        out = None
        out_lis = []
        for i in range(self.num_layers):
            out, hidden = model[i](x_input.float(),hidden.float())
            x_input = out
            out_lis.append(out)
        return out, hidden, out_lis
    
def diffusion_loss_fn(model,x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps):
    """对任意时刻t进行采样计算loss"""
    batch_size = x_0.shape[0]

    #对一个batchsize样本生成随机的时刻t
    t = torch.randint(0,n_steps,size=(batch_size//2,)).to(device)
    t = torch.cat([t,n_steps-1-t],dim=0)
    t = t.unsqueeze(-1)

    #x0的系数
    a = alphas_bar_sqrt[t].to(device)

    #eps的系数
    aml = one_minus_alphas_bar_sqrt[t].to(device)

    #生成随机噪音eps
    e = torch.randn_like(x_0).to(device)

    #构造模型的输入
    x = x_0*a+e*aml

    #送入模型，得到t时刻的随机噪声预测值
    output = model(x,t.squeeze(-1)).to(device)

    #与真实噪声一起计算误差，求平均值
    return (e - output).square().mean()


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
    def forward(self, label, mu, sigma,diff_loss):
        distribution = torch.distributions.cauchy.Cauchy(mu,sigma)
        loss = distribution.log_prob(label)
        return (-torch.mean(loss)+diff_loss)

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
    distribution = torch.distributions.cauchy.Cauchy(torch.Tensor(mu),torch.Tensor(sigma))
    sample = distribution.sample([10000])
    return(np.mean(ps.crps_ensemble(true,sample.permute(1,0))))
    #return(np.mean(ps.crps_gaussian(true,mu,sigma)))

def eval_pinball(true,mu,sigma,quantiles):
    losses = []
    true = torch.Tensor(true)
    mu = torch.Tensor(mu)
    sigma = torch.Tensor(sigma)
    for k in range(len(true)):
        ppf_list = cauchy.ppf(q = quantiles,loc = mu[k],scale = sigma[k])
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


