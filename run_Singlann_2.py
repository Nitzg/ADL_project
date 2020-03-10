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
import shutil

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x -0.5) *2
    return out.clamp(-1, 1)

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t


def add_noise(z,noiseAmp, is_cuda):
    if not is_cuda:
      noise = glo.generate_noise([z.shape[1],z.shape[2],z.shape[3]])
    else:
      noise = glo.generate_noise([z.shape[1],z.shape[2],z.shape[3]], device='cuda')
    return noise*noiseAmp

def round_down(num, divisor):
    return num - (num%divisor)

def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])

    inp = np.clip(inp,0,1)
    return inp

def save_images(number_of_images, G_nets, path_to_folder, is_cuda, reals):
  rounds = number_of_images//8
  for i in range(rounds):
    if is_cuda:
        z = torch.randn(8, 256).cuda()
    else:
        z = torch.randn(8, 256)
    for k in range(len(G_nets)-1):
        z = G_nets[k](z)
        z = z+add_noise(z,noiseAmp[k+1], is_cuda)
    images = G_nets[-1](z)
    for j in range(8):
        vutils.save_image(images[j],
                  'runs/image_as_input_GLO/samples/%s.png' % (str((i*8)+j)),
                  normalize=False)
  for i in range(number_of_images):
    plt.imsave(f'runs/image_as_input_GLO/actual/{i}.png', convert_image_np(reals[-1]), vmin=0, vmax=1)

  

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

    with open("sinsin/configs/sinGLO.yaml", 'r') as f:
        params = yaml.load(f, Loader=Loader)   

    if not os.path.isdir("runs"):
            os.mkdir("runs")
    if not os.path.isdir("runs/image_as_input_GLO"):
            os.mkdir("runs/image_as_input_GLO")
    if not os.path.isdir("runs/image_as_input_GLO/samples"):
            os.mkdir("runs/image_as_input_GLO/samples")
    if not os.path.isdir("runs/image_as_input_GLO/actual"):
            os.mkdir("runs/image_as_input_GLO/actual")
    

    G_nets = []
    Z_nets = []
    net_T = []
    noiseAmp = []
    Z = None

    
    is_cuda = True

    # glo     
    rn = f"{data[0].shape}"
    shutil.rmtree("runs/image_as_input_GLO/ims_%s" %rn, ignore_errors=True)
    os.mkdir("runs/image_as_input_GLO/ims_%s" % rn)
    if not os.path.isdir("runs/image_as_input_GLO/nets_%s" % rn):
        os.mkdir("runs/image_as_input_GLO/nets_%s" % rn)
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
        
    nt = glo.GLOTrainer(data[0], glo_params, rn, is_cuda, None, None, 0)
    G, Z, noise_amp = nt.train_glo(glo_opt_params)
    G_nets.append(G)
    Z_nets.append(Z)
    noiseAmp.append(noise_amp)
    

    
    z = torch.randn(64, 256)
    if is_cuda:
      z = z.cuda()
    print("shape of z")
    print(z.shape)
    ims = G(z)
    print(ims.shape)
    vutils.save_image(ims,'runs/image_as_input_GLO/ims_%s/samples.png' % (rn),normalize=False)
    net_T.append(Z)

    for i in range(1,len(data)):     
      rn = f"{data[i].shape}"
      shutil.rmtree("runs/image_as_input_GLO/ims_%s" % rn, ignore_errors=True)
      os.mkdir("runs/image_as_input_GLO/ims_%s" % rn)
      if not os.path.isdir("runs/image_as_input_GLO/nets_%s" % rn):
        os.mkdir("runs/image_as_input_GLO/nets_%s" %rn)
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
      


      nt = glo.GLOTrainer(data[i], glo_params, rn, is_cuda, net_T[-1], G_nets, i)
      G, Z, noise_amp = nt.train_glo(glo_opt_params)
      G_nets.append(G)
      noiseAmp.append(noise_amp)

      

      if is_cuda:
        z = torch.randn(8, 256).cuda()
        #z = net_T[-1](torch.randn(64, 32).cuda())
      else:
        z = torch.randn(8, 256)
        #z = net_T[-1](torch.randn(64, 32))
      for i in range(len(G_nets)-1):
        z = G_nets[i](z)
        z = z+add_noise(z,noiseAmp[i+1], is_cuda)
        #z = z.reshape(64, z.shape[1]*z.shape[2]*z.shape[3]) 
      vutils.save_image(G(z),
                  'runs/image_as_input_GLO/ims_%s/samples.png' % (rn),
                  normalize=False)
    path_to_folder = "runs/image_as_input_GLO/samples"
    save_images(1024, G_nets, path_to_folder, is_cuda, reals)



    
    