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

def add_noise(z,noiseAmp):
    noise = glo.generate_noise([z.shape[1],z.shape[2],z.shape[3]])
    return noise*noiseAmp

def round_down(num, divisor):
    return num - (num%divisor)

if __name__ == '__main__':
    # get scaled images
    reals = []
    for i in range(9):
        reals.append(torch.load(f"scaled_im_birds/{i}"))
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

    with open("sinsin/configs/Singlann.yaml", 'r') as f:
        params = yaml.load(f, Loader=Loader)   

    G_nets = []
    Z_nets = []
    net_T = []
    noiseAmp = []
    Z = None

    
    is_cuda = False

    # glo     
    rn = f"image_as_input_GLO{data[0].shape}"
    decay = params['glo']['decay']
    total_epoch = params['glo']['total_epoch']
    lr = params['glo']['learning_rate']
    factor = params['glo']['factor']
    nz = 256 #params['glo']['nz']
    batch_size = params['glo']['batch_size']
    sz = data[0].shape[2:4]
    glo_params = utils.GLOParams(nz=nz, do_bn=False, force_l2=False)
    glo_opt_params = utils.OptParams(lr=lr, factor=factor, batch_size=batch_size, epochs=total_epoch,
                                 decay_epochs=decay, decay_rate=0.5,  pad_image=[0,0])

    # add output of z to the noise
        
    nt = glo.GLOTrainer(data[0], glo_params, rn, is_cuda, None, None)
    G, Z, noise_amp = nt.train_glo(glo_opt_params)
    G_nets.append(G)
    Z_nets.append(Z)
    noiseAmp.append(noise_amp)
    
    # icp

    dim = params['icp']['dim']
    nepoch = params['icp']['total_epoch']

    if dim and nepoch:

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
    else:
      z = Z(torch.randn(64, 32))
      ims = netG(z)
      vutils.save_image(ims,
                  'runs/ims_%s/samples.png' % (rn),
                  normalize=False)
      net_T.append(Z)



    for i in range(1,len(data)):
      
      rn = f"image_as_input_GLO{data[i].shape}"
      decay = params['glo']['decay']
      total_epoch = params['glo']['total_epoch']
      lr = params['glo']['learning_rate']
      factor = params['glo']['factor']
      nz = data[i-1].shape[1:]#data[i-1].shape[1] * data[i-1].shape[2] *data[i-1].shape[3]
      batch_size = params['glo']['batch_size']
      sz = data[i].shape[2:4]
      pad_image = [(data[i].shape[2] - data[i-1].shape[2])/2, (data[i].shape[3] - data[i-1].shape[3])/2]
      #print(data[i].shape, data[i-1].shape)
      #print(pad_image)
      glo_params = utils.GLOParams(nz=nz, do_bn=False, force_l2=False)
      glo_opt_params = utils.OptParams(lr=lr, factor=factor, batch_size=batch_size, epochs=total_epoch,
                                 decay_epochs=decay, decay_rate=0.5,  pad_image= pad_image)
      


      nt = glo.GLOTrainer(data[i], glo_params, rn, is_cuda, net_T[-1], G_nets.copy())
      G, Z, noise_amp = nt.train_glo(glo_opt_params)
      G_nets.append(G)
      noiseAmp.append(noise_amp)

      

      if is_cuda:
        z = icpt.icp.netT(torch.randn(64, dim).cuda())
      else:
        z = icpt.icp.netT(torch.randn(64, dim))
      for i in range(len(G_nets)-1):
        z = G_nets[i](z)
        z = z+add_noise(z,noiseAmp[i+1])
        #z = z.reshape(64, z.shape[1]*z.shape[2]*z.shape[3]) 
      vutils.save_image(G(z),
                  'runs/ims_%s/samples.png' % (rn),
                  normalize=False)
  
    
    