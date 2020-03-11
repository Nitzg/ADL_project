from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import shutil
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F 
import torchvision.utils as vutils
import model
import utils


class GLO():
    def __init__(self, glo_params, image_params, rn, is_cuda, net_T, prev_G, scale):
        self.is_cuda = is_cuda
        self.vis_n = 64
        self.scale = scale 
        if net_T:
            self.netZ = net_T
            self.isT = True
            self.netG = model._netG_conv(glo_params.nz, image_params.sz, image_params.nc, glo_params.do_bn)
            fixed_noise = torch.FloatTensor(self.vis_n, glo_params.nz[0]*glo_params.nz[1]*glo_params.nz[2]).normal_(0, 1).reshape((self.vis_n, glo_params.nz[0],glo_params.nz[1],glo_params.nz[2]))

        else:
            self.netZ = model._netZ(glo_params.nz, image_params.n)
            self.netZ.apply(model.weights_init)
            self.isT = False
            #self.netZ = torch.randn(1, glo_params.nz)
            self.netG = model._netG(glo_params.nz, image_params.sz, image_params.nc, glo_params.do_bn)
            fixed_noise = torch.FloatTensor(self.vis_n, glo_params.nz).normal_(0, 1)
        self.netG.apply(model.weights_init)
        if self.is_cuda:
            self.netZ.cuda()
            self.netG.cuda()
        self.rn = rn
        

        

        self.fixed_noise = fixed_noise

        
        if self.is_cuda:
            self.fixed_noise.cuda()

        self.glo_params = glo_params
        self.image_params = image_params
        self.prev_G = prev_G

        self.dist = utils.distance_metric(image_params.sz[0], image_params.nc,
                                          glo_params.force_l2, is_cuda)
        self.l2Dist = nn.L1Loss()
        if is_cuda:
          self.l2Dist = self.l2Dist.cuda()
        self.noise_amp = 0

        
          


    def train(self, ims_np, opt_params, vis_epochs=1):

        # set RMSE and noiseAmp
        if self.isT:
          pad_image = opt_params.pad_image
          print(pad_image)
          m_image = nn.ZeroPad2d((int(pad_image[1]), int(pad_image[1]), int(pad_image[0]), int(pad_image[0])))
          zi = torch.randn(1, 64)
          if self.is_cuda:
              zi = zi.cuda()#.cuda()#self.netZ(torch.randn(1, 32).cuda())
          #zi_reshaped = zi
          for netG in self.prev_G[:self.scale]:
            zi = netG(zi)#_reshaped)
            #zi_reshaped = zi.reshape(1,zi.shape[1]*zi.shape[2]*zi.shape[3])
          criterion = nn.MSELoss()
          print("shape of image, m_image")
          if self.is_cuda:
            image = torch.from_numpy(ims_np[0]).cuda().view(1,3, ims_np[0].shape[1],ims_np[0].shape[2])
          else:
            image = torch.from_numpy(ims_np[0]).view(1,3, ims_np[0].shape[1],ims_np[0].shape[2])
          print(image.shape, m_image(zi).shape)
          RMSE = torch.sqrt(criterion(image, m_image(zi)))
          print("RMSE: ", RMSE)
          self.noise_amp = 1.0*RMSE
                
        for epoch in range(opt_params.epochs):
            er = self.train_epoch(ims_np, epoch, opt_params)
            print("NAG Epoch: %d Error: %f" % (epoch, er))
            #torch.save(self.netZ.state_dict(), 'runs/image_as_input_GLO/nets_%s/netZ_nag.pth' % self.rn)
            #torch.save(self.netG.state_dict(), 'runs/image_as_input_GLO/nets_%s/netG_nag.pth' % self.rn)
            #if epoch % vis_epochs == 0 and epoch >1 and not self.isT:
                #self.visualize(epoch, ims_np)
        
            
                
        return self.netG, self.netZ, self.noise_amp

    def train_epoch(self, ims_np, epoch, opt_params):
        batch_size =  1 #opt_params.batch_size
        batch_n = 1 #self.image_params.n // batch_size
        # Compute learning rate
        decay_steps = epoch // opt_params.decay_epochs
        lr = opt_params.lr * opt_params.decay_rate ** decay_steps
        # Initialize optimizers
        optimizerG = optim.Adam(self.netG.parameters(), lr=lr * opt_params.factor,
                                betas=(0.5, 0.999))
        optimizerZ = optim.Adam(self.netZ.parameters(), lr=lr,
                                betas=(0.5, 0.999))
        # Start optimizing
        er = 0
        for i in range(batch_n):
            # Put numpy data into tensors
            np_idx = np.array([0]) #rp[i * batch_size: (i + 1) * batch_size]
            idx = torch.from_numpy(np_idx).long() # .cuda()
            if self.is_cuda:
                idx = idx.cuda()
            np_data = ims_np #ims_np[rp[i * batch_size: (i + 1) * batch_size]]
            images = torch.from_numpy(np_data).float() #.cuda()
            if self.is_cuda:
                images = images.cuda()
            image = images[1].view(1,3, images.shape[2],images.shape[3])
            # Forward pass
            self.netZ.zero_grad()
            self.netG.zero_grad()
            if not self.isT:
                zi = torch.randn(1, 64)
                if self.is_cuda:
                    zi = zi.cuda()#.cuda()
                Ii = self.netG(zi.reshape(batch_size,self.glo_params.nz))

            else:
                zi = torch.randn(1, 64)#.cuda()#self.netZ(torch.randn(1, 32).cuda())
                if self.is_cuda:
                    zi = zi.cuda()#.cuda()
                #zi_reshaped = zi
                for pr_G in self.prev_G[:self.scale]:
                    zi = pr_G(zi)#_reshaped)
                    #zi_reshaped = zi.reshape(1,zi.shape[1]*zi.shape[2]*zi.shape[3])
                if self.is_cuda:
                  noise_ = utils.generate_noise([zi.shape[1],zi.shape[2],zi.shape[3]], device = 'cuda')
                else:
                  noise_ = utils.generate_noise([zi.shape[1],zi.shape[2],zi.shape[3]])
                zi = self.noise_amp*noise_+zi
                Ii = self.netG(zi.reshape(batch_size,zi.shape[1],zi.shape[2],zi.shape[3]))
            rec_loss = 1*self.l2Dist(2 * Ii - 1, 2 * image - 1) + 0 *self.dist(2 * Ii - 1, 2 * image - 1) 
            #print(rec_loss)
            # Backward pass and optimization step
            rec_loss.backward(retain_graph=True)
            optimizerG.step()
            if not self.isT:
              optimizerZ.step()
            er += rec_loss.item()
        if not self.isT:
          self.netZ.get_norm()
        er = er / batch_n
        return er




class GLOTrainer():
    def __init__(self, ims_np, glo_params, rn, is_cuda, net_T, prev_G, scale):
        self.ims_np = ims_np
        self.sz = ims_np.shape[2:4]
        self.rn = rn
        self.nc = ims_np.shape[1]
        self.n = ims_np.shape[0]
        self.scale = scale
        self.image_params = utils.ImageParams(sz=self.sz, nc=self.nc, n=self.n)
        self.glo = GLO(glo_params, self.image_params, rn, is_cuda, net_T, prev_G, scale)
        

    def train_glo(self, opt_params):
        G, Z, noise_amp = self.glo.train(self.ims_np, opt_params, vis_epochs= 20)
        return G, Z, noise_amp
