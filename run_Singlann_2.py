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



def get_scaled_images(scale,input_folder_path ):
    reals = []
    for i in range(scale):
        reals.append(torch.load(f"{input_folder_path}/{i}"))
        max_dim = utils.round_down(reals[i].shape[3], 4) 
        reals[i] = reals[i][:,:,0:max_dim, 0:max_dim]
        print(reals[i].shape)
    print(len(reals))

    data = []

    for real in reals:
        data.append(np.vstack([real]*100))
        print(data[-1].shape) 
    return reals, data       
         


def run_for_image(scale,input_folder_path, output_folder_name ):

    # params:
    scale = scale
    input_folder_path = input_folder_path
    output_folder_name = output_folder_name
    


    if not os.path.isdir("runs"):
            os.mkdir("runs")
    if not os.path.isdir(f"runs/{output_folder_name}"):
            os.mkdir(f"runs/{output_folder_name}")
    
    
    is_cuda = False
    decay = 20
    total_epoch = 600
    lr = 0.01
    factor = 0.1
    batch_size = 1
    G_nets = []
    Z_nets = []
    net_T = []
    noiseAmp = []
    Z = None
    
    reals,data = get_scaled_images(scale, input_folder_path)

    nz = 64 
    nz_first = 64
    sz = data[0].shape[2:4]
 
    rn = f"{data[0].shape}"
    shutil.rmtree(f"runs/{output_folder_name}/ims_%s" %rn, ignore_errors=True)
    os.mkdir(f"runs/{output_folder_name}/ims_%s" % rn)
    if not os.path.isdir(f"runs/{output_folder_name}/nets_%s" % rn):
        os.mkdir(f"runs/{output_folder_name}/nets_%s" % rn)
    
    
    glo_params = utils.GLOParams(nz=nz, do_bn=False, force_l2=False)
    glo_opt_params = utils.OptParams(lr=lr, factor=factor, batch_size=batch_size, epochs=total_epoch,
                                 decay_epochs=decay, decay_rate=0.5,  pad_image=[0,0])

    
        
    nt = glo.GLOTrainer(data[0], glo_params, rn, is_cuda, None, None, 0)
    G, Z, noise_amp = nt.train_glo(glo_opt_params)
    G_nets.append(G)
    Z_nets.append(Z)
    noiseAmp.append(noise_amp)
    

    
    z = torch.randn(64, nz_first)
    if is_cuda:
      z = z.cuda()
    print("shape of z")
    print(z.shape)
    ims = G(z)
    print(ims.shape)
    vutils.save_image(ims,f"runs/{output_folder_name}/ims_%s/samples.png" % (rn),normalize=False)
    net_T.append(Z)
    path_to_folder = f"runs/{output_folder_name}/ims_%s"% (rn)
    utils.save_images(512, G_nets, noiseAmp , path_to_folder, is_cuda, reals[0])

    for i in range(1,len(data)):     
      rn = f"{data[i].shape}"
      shutil.rmtree(f"runs/{output_folder_name}/ims_%s" % rn, ignore_errors=True)
      os.mkdir(f"runs/{output_folder_name}/ims_%s" % rn)
      if not os.path.isdir(f"runs/{output_folder_name}/nets_%s" % rn):
        os.mkdir(f"runs/{output_folder_name}/nets_%s" %rn)
      nz = data[i-1].shape[1:]
      sz = data[i].shape[2:4]
      pad_image = [(data[i].shape[2] - data[i-1].shape[2])/2, (data[i].shape[3] - data[i-1].shape[3])/2]
      glo_params = utils.GLOParams(nz=nz, do_bn=False, force_l2=False)
      glo_opt_params = utils.OptParams(lr=lr, factor=factor, batch_size=batch_size, epochs=total_epoch,
                                 decay_epochs=decay, decay_rate=0.5,  pad_image= pad_image)


      nt = glo.GLOTrainer(data[i], glo_params, rn, is_cuda, net_T[-1], G_nets, i)
      G, Z, noise_amp = nt.train_glo(glo_opt_params)
      G_nets.append(G)
      noiseAmp.append(noise_amp)

      

      if is_cuda:
        z = torch.randn(8, nz_first).cuda()
      else:
        z = torch.randn(8, nz_first)
      for i in range(len(G_nets)-1):
        z = G_nets[i](z)
        z = z+utils.add_noise(z,noiseAmp[i+1], is_cuda)
      vutils.save_image(G(z),
                  f"runs/{output_folder_name}/ims_%s/samples.png" % (rn),
                  normalize=False)
      path_to_folder = f"runs/{output_folder_name}/ims_%s"% (rn)
      utils.save_images(512, G_nets, noiseAmp , path_to_folder, is_cuda, reals[i])


if __name__ == '__main__':
    scales = [4,4,4,7,7,7,9,9,9]
    input_folders = ["scaled_im_birds_4","scaled_im_mountains3_4", "scaled_im_starry_night_4", "scaled_im_birds_7","scaled_im_mountains3_7" ,"scaled_im_starry_night_7","scaled_im_birds_9", "scaled_im_mountains3_9", "scaled_im_starry_night_9"]
    output_folder_names = ["scaled_im_birds_4","scaled_im_mountains3_4", "scaled_im_starry_night_4", "scaled_im_birds_7","scaled_im_mountains3_7" ,"scaled_im_starry_night_7","scaled_im_birds_9", "scaled_im_mountains3_9", "scaled_im_starry_night_9"]
    
    for i in range(len(scales)):
      run_for_image(scales[i], input_folders[i], output_folder_names[i])
