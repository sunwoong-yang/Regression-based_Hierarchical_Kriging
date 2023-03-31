import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler as Scaler

def Dat2Ten(dataloader):
    x_total, y_total = [], []
    for x,y in dataloader:
        x_total.append(x)
        y_total.append(y)
    return torch.cat(x_total, dim=0), torch.cat(y_total, dim=0)

def Dat2Num(dataloader):
    x_total, y_total = [], []
    for x,y in dataloader:
        x_total.append(x)
        y_total.append(y)
    return np.concatenate(x_total, axis=0), np.concatenate(y_total, axis=0)

def Num2Dat(X, Y, mini_batch = None):
    data = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)) # create your datset
    if mini_batch is None:
        mini_batch = len(X)
    return DataLoader(data, batch_size=mini_batch)

def Num2Ten(x):
        return torch.tensor(np.array(x), dtype=torch.float32)

def Ten2Dat(X, Y, mini_batch = None):
    data = TensorDataset(X,Y) # create your datset
    if mini_batch is None:
        mini_batch = len(X)
    return DataLoader(data, batch_size=mini_batch)

# def Ten2Num(x, detach=True):
#     if detach:
#         return x.detach().numpy()
#     else:
#         return x.numpy()

def Ten2Num(x, detach=True):
    if x.requires_grad:
        return x.detach().numpy()
    else:
        return x.numpy()

def NLLloss(y_real, y_pred, var):
    return (torch.log(var) + ((y_real - y_pred).pow(2))/var).mean()/2 + 0.5*np.log10(2*np.pi)

def read_csv(N_inp, dir="DOEset1.csv"):
    data = pd.read_csv(dir)
    header = list(data.columns)
    inp_dataset = data[header[:N_inp]].values
    out_dataset = data[header[N_inp:]].values

    return header[:N_inp], header[N_inp:], inp_dataset, out_dataset

def csv2Dat(N_inp, dir="DOEset1.csv", mini_batch = None):
    H_inp, H_out, X, Y = read_csv(N_inp, dir)
    return H_inp, H_out, Num2Dat(X, Y, mini_batch)

def csv2Ten(N_inp, dir="DOEset1.csv", mini_batch = None):
    H_inp, H_out, X, Y = read_csv(N_inp, dir)
    return H_inp, H_out, Num2Ten(X), Num2Ten(Y)

def csv2Num(N_inp, dir="DOEset1.csv", mini_batch = None):
    H_inp, H_out, inp_dataset, out_dataset =  read_csv(N_inp, dir)
    return H_inp, H_out, inp_dataset, out_dataset

def normalize(data):
    STD = Scaler()
    scaled_data = STD.fit_transform(data)
    return scaled_data, STD
