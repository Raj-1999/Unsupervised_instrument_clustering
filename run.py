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

from model import *
from dataset import *
from utils import progress_bar

import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import pylab
import matplotlib

parser=argparse.ArgumentParser(description="Unsupervised Instrument Clustering")
parser.add_argument('--lr',default=1e-3,type=float,help='learning rate')
parser.add_argument('--epochs',type=int,default=200,help='No.of training epochs')
parser.add_argument('--batch_size',type=int,default=9)
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
    # F.pad(input = recon_x, pad = (0,1,0,1), mode = 'constant')
    target = torch.zeros(recon_x.shape[0],1,40,40)
    target = target.to(device)
    source = recon_x
    target[:,:, :38, :] = source
    MSE = F.mse_loss(target,x)
    #BCE=F.binary_cross_entropy(recon_x, x.view(x.size(0),-1), size_average=False)
    
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD

print("PREPARING DATA")

criterion=nn.MSELoss()

print('CREATING NETWORK')
vae=VAE().to(device)


print('LOADING DATA')
train_set=InstrumentsDataset()
test_set=InstrumentsDataset(test=True)


def train_instruments(currentepoch,epoch):
    dataloader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    dataloader=iter(dataloader)
    print('\n=> Instrument Epoch: %d' % currentepoch)
    train_loss,total,correct=0,0,0
    params = vae.parameters()
    optimizer=optim.Adam(vae.parameters(),lr=args.lr)
    
    for batch_idx in range(len(dataloader)):
        inputs=next(dataloader)
        #input=input.view(input.size(0),input.size(1),)
        #inputs= torch.tensor(inputs).type(torch.FloatTensor)
        inputs=inputs.clone().detach().type(torch.FloatTensor)
        # if(np.shape(inputs)[0]!=16):
        #     continue

        inputs= inputs.to(device)
        
        optimizer.zero_grad()

        true_samples = Variable(torch.randn(64, 2000),requires_grad=False)
        true_samples = true_samples.to(device)
        y_pred, mu, log_var = vae(inputs)
        #y_pred, z = vae(inputs)
        #y_pred = vae(inputs)
        #print(y_pred[0].shape)
        #print(inputs[0])
        # ------------------------------------------------------
        save_path='/home/nevronas/Projects/Nevronas-Projects/Audio/Unsupervised_instrument_clustering/plots/output.jpg'
        pylab.axis('off')
        pylab.axes([0.,0.,1.,1.], frameon=True, xticks=[], yticks=[])
        # print(np.shape(y_pred))
        # print(y_pred[0].detach().cpu().numpy().shape)
        librosa.display.specshow(librosa.power_to_db(y_pred[0].detach().cpu().numpy()[0]),x_axis='time')
        pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
        pylab.close()
        # ---------------------------------------------------
                # ------------------------------------------------------
        save_path='/home/nevronas/Projects/Nevronas-Projects/Audio/Unsupervised_instrument_clustering/plots/input.jpg'
        pylab.axis('off')
        pylab.axes([0.,0.,1.,1.], frameon=True, xticks=[], yticks=[])
        # print(np.shape(y_pred))
        # print(y_pred[0].detach().cpu().numpy().shape)
        librosa.display.specshow(librosa.power_to_db(inputs[0].detach().cpu().numpy()[0]),x_axis='time')
        pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
        pylab.close()
        # ---------------------------------------------------
        #loss = F.mse_loss(y_pred, inputs[:, :, 0:127, 0:127]) #compute_mmd(true_samples, z) + F.mse_loss(y_pred, inputs[:, :, 0:127, 0:127])
        loss = loss_fn(y_pred,inputs,mu,log_var)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        #-,predict=y_pred.max(1)
        total += inputs.size(0)
        #correct+=predict.eq(inputs).sum().item()
        
        with open("./logs/instrument_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / total))

       # with open("./logs/instrument_train_acc.log", "a+") as afile:
        #    afile.write("{}\n".format(correct / total))

        del inputs
        gc.collect()
        torch.cuda.empty_cache()
        torch.save(vae.state_dict(), './weights/network_train.ckpt')
        with open("./information/instrument_info.txt", "w+") as f:
            f.write("{} {}".format(currentepoch, batch_idx))
        print('Batch: [%d/%d], Loss: %.3f, Train Loss: %.3f , %d' % (batch_idx, len(dataloader), loss.item(), train_loss/(batch_idx+1), total), end='\r')

    torch.save(vae.state_dict(), './checkpoints/network_train_epoch_{}.ckpt'.format(currentepoch + 1))
    print('=> Classifier Network : Epoch [{}/{}], Loss:{:.4f}'.format(currentepoch+1, epoch, train_loss / len(dataloader)))

def test_instruments(currentepoch, epoch):
    dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)
    print('\n=> Instrument Testing Epoch: %d' % currentepoch)
    
    test_loss, correct, total = 0, 0, 0

    for batch_idx in range(len(dataloader)):
        inputs = next(dataloader)
        inputs = torch.tensor(inputs).type(torch.FloatTensor)
        inputs = inputs.to(device)
        y_pred,mu,log_var = vae(inputs)

        loss = loss_fn(y_pred, inputs,mu,log_var)
        test_loss+=loss.item()
        #-,predict=y_pred.max(1)
        total+=inputs.size(0)
       # correct+=predict.eq(inputs).sum().item()

         
        with open("./logs/instrument_test_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(test_loss / total))

        #with open("./logs/instrument_test_acc.log", "a+") as afile:
            #afile.write("{}\n".format(correct / total))

        del inputs
        gc.collect()
        torch.cuda.empty_cache()
        print('Batch: [%d/%d], Loss: %.3f, Train Loss: %.3f ,%d' % (batch_idx, len(dataloader), loss.item(), test_loss/(batch_idx+1), total), end='\r')

    print('=> Classifier Network Test: Epoch [{}/{}], Loss:{:.4f}'.format(currentepoch+1, epoch, test_loss / len(dataloader)))

print('==> Training starts..')
for epoch in range(args.epochs):
    train_instruments(epoch, args.epochs)
    test_instruments(epoch, args.epochs)
