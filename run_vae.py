import os
import gc
import torch
import argparse
import librosa
import matplotlib
matplotlib.use('Agg')
import numpy as np
from collections import Counter
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader 
from torchvision.utils import save_image

import librosa.display

from vae import *
from dataset import *
from utils import progress_bar
from torch.optim.lr_scheduler import *

import matplotlib.pyplot as plt
import pylab
import matplotlib

parser=argparse.ArgumentParser(description="Unsupervised Instrument Clustering")
parser.add_argument('--lr',default=1e-4, type=float, help='learning rate')
parser.add_argument('--epochs',type=int,default=400,help='No.of training epochs')
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--prepare_data',type=int,default=1)

args=parser.parse_args()
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def loss_fn(recon_x, x, mu, logvar):
    #MSE = F.mse_loss(recon_x, x, size_average=False)
    MSE = F.binary_cross_entropy(F.sigmoid(recon_x), F.sigmoid(x), size_average=False)
    #KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = torch.sum(0.5 * (mu ** 2 + torch.exp(logvar) - logvar - 1))
    return MSE + KLD, MSE, KLD

criterion=nn.MSELoss()

print('CREATING NETWORK')
vae = VAE().to(device)


print('LOADING DATA')
train_set = InstrumentsDatasetVAE()

optimizer = optim.Adam(vae.parameters(),
                           lr=args.lr,
                           betas=(0.99,0.95),
                           eps=1e-8,
                           weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

def train_instruments(currentepoch, epoch):
    dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)
    print('\n=> Instrument Epoch: %d' % currentepoch)
    train_loss, total = 0, 0
    scheduler.step()
    
    for batch_idx in range(len(dataloader)):
        inputs = next(dataloader)
        inputs = torch.tensor(inputs).type(torch.FloatTensor)

        inputs = inputs.to(device)
        
        optimizer.zero_grad()

        y_pred, mu, log_var, _ = vae(inputs)
    
        loss, mse, kld = loss_fn(y_pred, inputs, mu, log_var)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 0.25)
        optimizer.step()
        
        train_loss += loss.item()
        total += inputs.size(0)
        
        with open("./logs/instrumentsvae_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / total))

        del inputs
        gc.collect()
        torch.cuda.empty_cache()
        torch.save(vae.state_dict(), './weights/networkvae_train.ckpt')
        with open("./information/instrumentsvae_info.txt", "w+") as f:
            f.write("{} {}".format(currentepoch, batch_idx))
        print('Batch: [%d/%d], Loss: %.3f, Train Loss: %.3f , MSE Loss: %.3f , KLD Loss: %.3f ,%d' % (batch_idx, len(dataloader), loss.item(), train_loss/(batch_idx+1), mse.item(), kld.item(), total), end='\r')

    torch.save(vae.state_dict(), './checkpoints/networkvae_train_epoch_{}.ckpt'.format(currentepoch + 1))
    print('\n=> Network : Epoch [{}/{}], Loss:{:.4f}'.format(currentepoch+1, epoch, train_loss / len(dataloader)))


print('==> Training starts..')
for epoch in range(args.epochs):
    train_instruments(epoch, args.epochs)
