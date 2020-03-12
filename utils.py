from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import sys
import numpy as np
import torch
import torch.nn as nn
import vgg_metric
import shutil
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x -0.5) *2
    return out.clamp(-1, 1)

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)


def generate_noise(size,num_samp=1,device='cpu',type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise,size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise


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

    inp = np.clip(inp,0,1)
    return inp

def add_noise(z,noiseAmp, is_cuda):
    if not is_cuda:
      noise = generate_noise([z.shape[1],z.shape[2],z.shape[3]])
    else:
      noise = generate_noise([z.shape[1],z.shape[2],z.shape[3]], device='cuda')
    return noise*noiseAmp

def save_images(number_of_images, G_nets, noiseAmp, path_to_folder, is_cuda, reals):
  if not os.path.isdir(f"{path_to_folder}/samples"):
            os.mkdir(f"{path_to_folder}/samples")
  if not os.path.isdir(f"{path_to_folder}/actual"):
            os.mkdir(f"{path_to_folder}/actual")
  rounds = number_of_images//8
  for i in range(rounds):
    if is_cuda:
        z = torch.randn(8, 64).cuda()
    else:
        z = torch.randn(8, 64)
    for k in range(len(G_nets)-1):
        z = G_nets[k](z)
    images = G_nets[-1](z)
    for j in range(8):
        vutils.save_image(images[j],
                  f'{path_to_folder}/samples/%s.png' % (str((i*8)+j)),
                  normalize=False)
  for i in range(number_of_images):
    plt.imsave(f'{path_to_folder}/actual/{i}.png', convert_image_np(reals), vmin=0, vmax=1)

  


GParams = collections.namedtuple('GParams', 'nz ngf do_bn mu sd force_l2')
GParams.__new__.__defaults__ = (None, None, None, None, None, None)
OptParams = collections.namedtuple('OptParams', 'lr factor ' +
                                                'batch_size epochs ' +
                                                'decay_epochs decay_rate lr_ratio pad_image')
OptParams.__new__.__defaults__ = (None, None, None, None, None, None)
ImageParams = collections.namedtuple('ImageParams', 'sz nc n mu sd')
ImageParams.__new__.__defaults__ = (None, None, None)


def distance_metric(sz, nc, force_l2=False, is_cuda = True):
    if force_l2:
        return nn.L1Loss()
    if sz == 24:
        return vgg_metric._VGGDistance(2, is_cuda)
    elif sz > 24  and sz <= 32:
        return vgg_metric._VGGDistance(3, is_cuda)
    elif sz > 32 and sz <= 64:
        return vgg_metric._VGGDistance(4, is_cuda)
    elif sz > 64:
        return vgg_metric._VGGMSDistance(is_cuda)


def sample_gaussian(x, m, is_cuda):
    x = x.data.numpy()
    mu = x.mean(0).squeeze()
    cov2 = np.cov(x, rowvar=0)
    z = np.random.multivariate_normal(mu, cov2, size=m)
    z_t = torch.from_numpy(z).float()
    radius = z_t.norm(2, 1).unsqueeze(1).expand_as(z_t)
    z_t = z_t / radius
    if is_cuda:
      return z_t.cuda()
    else:
      return z_t


def unnorm(ims, mu, sd):
    for i in range(len(mu)):
        ims[:, i] = ims[:, i] * sd[i]
        ims[:, i] = ims[:, i] + mu[i]
    return ims


def format_im(ims_gen, mu, sd):
    if ims_gen.size(1) == 3:
        rev_idx = torch.LongTensor([2, 1, 0])#.cuda()
    elif ims_gen.size(1) == 1:
        rev_idx = torch.LongTensor([0])#.cuda()
    else:
        arr = [i for i in range(ims_gen.size(1))]
        rev_idx = torch.LongTensor(arr)#.cuda()
    # Generated images
    ims_gen = unnorm(ims_gen, mu, sd)
    ims_gen = ims_gen.data.index_select(1, rev_idx)
    ims_gen = torch.clamp(ims_gen, 0, 1)
    return ims_gen
