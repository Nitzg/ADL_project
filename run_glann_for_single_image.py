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

    real = torch.load(f"scaled_im_birds/8")
    #max_dim = round_down(reals[i].shape[3], 4) 
    #reals[i] = reals[i][:,:,0:max_dim, 0:max_dim]
    print(real.shape)
    

    data = np.vstack([real]*100)
            
    try:
        from yaml import CLoader as Loader, CDumper as Dumper
    except ImportError:
        from yaml import Loader, Dumper

    with open("sinsin/configs/glann_for_single_image.yaml", 'r') as f:
        params = yaml.load(f, Loader=Loader)   

    # glo
         
    rn = f"glann_for_single_image{data.shape}"
    decay = params['glo']['decay']
    total_epoch = params['glo']['total_epoch']
    lr = params['glo']['learning_rate']
    factor = params['glo']['factor']
    nz = params['glo']['nz']
    batch_size = params['glo']['batch_size']
    sz = data.shape[2:4]
    is_cuda = params['glo']['is_cuda']
    glo_params = utils.GLOParams(nz=nz, do_bn=False, force_l2=False)
    glo_opt_params = utils.OptParams(lr=lr, factor=factor, batch_size=batch_size, epochs=total_epoch,
                                decay_epochs=decay, decay_rate=0.5)

    
    nt = glo.GLOTrainer(data, glo_params, rn, is_cuda,[], [])
    G, Z = nt.train_glo(glo_opt_params)
    

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
    
    #net_T.append(icpt.icp.netT)
    print("shape of z")
    print(z.shape)
    ims = netG(z)
    print(ims.shape)
    vutils.save_image(ims,
                'runs/ims_%s/samples.png' % (rn),
                normalize=False)

            

    