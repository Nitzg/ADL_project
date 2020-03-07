from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Conv') != -1:
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Emb') != -1:
        init.normal_(m.weight, mean=0, std=0.01)


class _netZ(nn.Module):
    def __init__(self, nz, n):
        super(_netZ, self).__init__()
        self.n = n
        self.emb = nn.Embedding(self.n, nz)
        self.nz = nz
        
        

    def get_norm(self):
        wn = self.emb.weight.norm(2, 1).data.unsqueeze(1)
        self.emb.weight.data = \
            self.emb.weight.data.div(wn.expand_as(self.emb.weight.data))

    def forward(self, idx):
        z = self.emb(idx).squeeze() 
        return z


class _netG(nn.Module):
    def __init__(self, nz, sz, nc, do_bn=False, prev_G = []):
        super(_netG, self).__init__()
        print(nz)
        self.sz = [sz[0], sz[1]]
        self.dim_im = 128 * (sz[0] // 4) * (sz[1] // 4)
        self.lin_in = nn.Linear(nz, 1024, bias=False)
        self.lin_im = nn.Linear(1024, self.dim_im, bias=False)

        self.conv1 = nn.ConvTranspose2d(128, nz, 4, 2, 1, bias=True)
        self.conv2 = nn.ConvTranspose2d(nz, nc, 4, 2, 1, bias=True)
        self.sig = nn.Sigmoid()
        self.do_bn = do_bn

        self.nonlin = nn.LeakyReLU(0.2, inplace=True)
        self.prev_G = prev_G

    def main(self, z):
        z = self.lin_in(z)
        z = self.nonlin(z)
        z = self.lin_im(z)
        z = self.nonlin(z)
        z = z.view(-1, 128, self.sz[0] // 4 , self.sz[1] // 4 )
        z = self.conv1(z)
        z = self.nonlin(z)
        z =  self.conv2(z)
        z = self.sig(z)
        return z

    def forward(self, z):
        zn = z.norm(2, 1).detach().unsqueeze(1).expand_as(z)
        z = z.div(zn)
        output = self.main(z)
        return output
