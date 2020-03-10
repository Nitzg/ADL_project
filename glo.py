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
    def __init__(self, glo_params, image_params, rn, is_cuda, net_T, prev_G):
        self.is_cuda = is_cuda
        self.vis_n = 64
        if net_T:
            self.netZ = net_T
            self.isT = True
            self.netG = model._netG_conv(glo_params.nz, image_params.sz, image_params.nc, glo_params.do_bn, prev_G)
            fixed_noise = torch.FloatTensor(self.vis_n, glo_params.nz[0]*glo_params.nz[1]*glo_params.nz[2]).normal_(0, 1).reshape((self.vis_n, glo_params.nz[0],glo_params.nz[1],glo_params.nz[2]))

        else:
            self.netZ = model._netZ(glo_params.nz, image_params.n)
            self.netZ.apply(model.weights_init)
            self.isT = False
            self.netG = model._netG(glo_params.nz, image_params.sz, image_params.nc, glo_params.do_bn, prev_G)
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
        prev_er = 100000

        # set RMSE and noiseAmp
        if self.isT:
          pad_image = opt_params.pad_image
          print(pad_image)
          m_image = nn.ZeroPad2d((int(pad_image[1]), int(pad_image[1]), int(pad_image[0]), int(pad_image[0])))
          #TODO change to cuda if cuda
          zi = self.netZ(torch.randn(1, 32))
          #zi_reshaped = zi
          for netG in self.prev_G:
            zi = netG(zi)#_reshaped)
            #zi_reshaped = zi.reshape(1,zi.shape[1]*zi.shape[2]*zi.shape[3])
          criterion = nn.MSELoss()
          print("shape of image, m_image")
          # image = torch.from_numpy(ims_np[0]).cuda().view(1,3, ims_np[0].shape[1],ims_np[0].shape[2])
          image = torch.from_numpy(ims_np[0]).view(1,3, ims_np[0].shape[1],ims_np[0].shape[2])
          print(image.shape, m_image(zi).shape)
          RMSE = torch.sqrt(criterion(image, m_image(zi)))
          print("RMSE: ", RMSE)
          self.noise_amp = 1.0*RMSE
                
        for epoch in range(opt_params.epochs):
            er = self.train_epoch(ims_np, epoch, opt_params)
            print("NAG Epoch: %d Error: %f" % (epoch, er))
            torch.save(self.netZ.state_dict(), 'runs/nets_%s/netZ_nag.pth' % self.rn)
            torch.save(self.netG.state_dict(), 'runs/nets_%s/netG_nag.pth' % self.rn)
            if epoch % vis_epochs == 0 and epoch >1 and not self.isT:
                self.visualize(epoch, ims_np)
        
            
                
        return self.netG, self.netZ, self.noise_amp

    def train_epoch(self, ims_np, epoch, opt_params):
        #rp = np.random.permutation(self.image_params.n)
        # Compute batch size
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
                zi = self.netZ(idx) 
                Ii = self.netG(zi.reshape(batch_size,self.glo_params.nz))

            else:
                zi = self.netZ(torch.randn(1, 32))#.cuda())
                #zi_reshaped = zi
                for pr_G in self.prev_G:
                    zi = pr_G(zi)#_reshaped)
                    #zi_reshaped = zi.reshape(1,zi.shape[1]*zi.shape[2]*zi.shape[3])
                noise_ = generate_noise([zi.shape[1],zi.shape[2],zi.shape[3]])
                zi = self.noise_amp*noise_+zi
                Ii = self.netG(zi.reshape(batch_size,zi.shape[1],zi.shape[2],zi.shape[3]))
            rec_loss = 0.5*self.l2Dist(2 * Ii - 1, 2 * image - 1) + 0.5 *self.dist(2 * Ii - 1, 2 * image - 1) 
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

    def visualize(self, epoch, ims_np):
        if self.is_cuda:
            Igen = self.netG(self.fixed_noise.cuda())
        else:
            Igen = self.netG(self.fixed_noise)
        z = utils.sample_gaussian(self.netZ.emb.weight.clone().cpu(),
                                  self.vis_n, self.is_cuda)
        Igauss = self.netG(z)
        idx = torch.from_numpy(np.arange(self.vis_n))#.cuda()
        if self.is_cuda:
            idx = idx.cuda()
        Irec = self.netG(self.netZ(idx))
        
        Iact = torch.from_numpy(ims_np[:self.vis_n])#.cuda()
        if self.is_cuda:
            Iact = Iact.cuda()

        #epoch = 0
        # Generated images
        vutils.save_image(Igen.data,
                          'runs/ims_%s/generations_epoch_%03d.png' % (self.rn, epoch),
                          normalize=False)
        # Reconstructed images
        vutils.save_image(Irec.data,
                          'runs/ims_%s/reconstructions_epoch_%03d.png' % (self.rn, epoch),
                          normalize=False)
    
        vutils.save_image(Iact.data,
                          'runs/ims_%s/act.png' % (self.rn),
                          normalize=False)
        vutils.save_image(Igauss.data,
                          'runs/ims_%s/gaussian_epoch_%03d.png' % (self.rn, epoch),
                          normalize=False)

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

class GLOTrainer():
    def __init__(self, ims_np, glo_params, rn, is_cuda, net_T, prev_G):
        self.ims_np = ims_np
        self.sz = ims_np.shape[2:4]
        self.rn = rn
        self.nc = ims_np.shape[1]
        self.n = ims_np.shape[0]
        self.image_params = utils.ImageParams(sz=self.sz, nc=self.nc, n=self.n)
        self.glo = GLO(glo_params, self.image_params, rn, is_cuda, net_T, prev_G)
        if not os.path.isdir("runs"):
            os.mkdir("runs")
        shutil.rmtree("runs/ims_%s" % self.rn, ignore_errors=True)
        # shutil.rmtree("nets", ignore_errors=True)
        os.mkdir("runs/ims_%s" % self.rn)
        if not os.path.isdir("runs/nets_%s" % self.rn):
            os.mkdir("runs/nets_%s" % self.rn)

    def train_glo(self, opt_params):
        G, Z, noise_amp = self.glo.train(self.ims_np, opt_params, vis_epochs= 20)
        return G, Z, noise_amp
