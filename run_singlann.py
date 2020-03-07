import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F 
import numpy as np
import argparse
import yaml
import torch
import glo
import imresize
import utils
import torchvision.utils as vutils
import os
import icp
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage import io as img
import model



def round_down(num, divisor):
    return num - (num%divisor)

if __name__ == '__main__':
    # get scaled images
    reals = []
    for i in range(9):
        reals.append(torch.load(f"scaled_im/{i}"))
        max_dim = round_down(reals[i].shape[3], 4) 
        reals[i] = reals[i][:,:,0:max_dim, 0:max_dim]
        print(reals[i].shape)
    print(len(reals))

    data = []

    for real in reals:
        data.append(np.vstack([real]*100))
        print(data[-1].shape)        
        
    try:
        from yaml import CLoader as Loader, CDumper as Dumper
    except ImportError:
        from yaml import Loader, Dumper

    with open("configs/Singlann.yaml", 'r') as f:
        params = yaml.load(f, Loader=Loader)   

    G_nets = []
    Z_nets = []
    net_T = []
    Z = None

    
    is_cuda = True

    # glo

    for i in range(len(data)):
        
        rn = f"same_Z_VGG_according_to_im_size_and_L2{data[i].shape}"
        decay = params['glo']['decay']
        total_epoch = 100
        lr = params['glo']['learning_rate']
        factor = params['glo']['factor']
        nz = 256 #params['glo']['nz']
        batch_size = params['glo']['batch_size']
        sz = data[i].shape[2:4]
        glo_params = utils.GLOParams(nz=nz, do_bn=False, force_l2=False)
        glo_opt_params = utils.OptParams(lr=lr, factor=factor, batch_size=batch_size, epochs=total_epoch,
                                 decay_epochs=decay, decay_rate=0.5)

        # add output of z to the noise
        
        nt = glo.GLOTrainer(data[i], glo_params, rn, is_cuda, Z_nets.copy(), G_nets.copy())
        G, Z = nt.train_glo(glo_opt_params)
        G_nets.append(G)
        Z_nets.append(Z)
        
        

        # icp

        dim = params['icp']['dim']
        nepoch = params['icp']['total_epoch']

        W = torch.load('runs/nets_%s/netZ_nag.pth' % (rn))
        W = W['emb.weight'].data.cpu().numpy()

        netG = model._netG(nz, sz, 3)
        if is_cuda:
            netG = netG.cuda()
        state_dict = torch.load('runs/nets_%s/netG_nag.pth' % (rn))
        netG.load_state_dict(state_dict)

        icpt = icp.ICPTrainer(W, dim, is_cuda)
        icpt.train_icp(nepoch)
        torch.save(icpt.icp.netT.state_dict(), 'runs/nets_%s/netT_nag.pth' % rn)

        if is_cuda:
          z = icpt.icp.netT(torch.randn(64, dim).cuda())
        else:
          z = icpt.icp.netT(torch.randn(64, dim))
        
        net_T.append(icpt.icp.netT)
        print("shape of z")
        print(z.shape)
        ims = netG(z)
        print(ims.shape)
        vutils.save_image(ims,
                  'runs/ims_%s/samples.png' % (rn),
                  normalize=False)

        #if i == 5:
        #    break
            

    