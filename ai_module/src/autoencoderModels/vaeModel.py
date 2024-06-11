import torch
import torch.nn as nn

class vaework12(nn.Module):
   def __init__(self, latdim):
       super().__init__()
       self.latdim = latdim

       self.l0_chan = 3
       self.l1_chan = 6
       self.l2_chan = 8
       self.l3_chan = 10
       self.l4_chan = 12
       self.l5_chan = 12
       self.h_neun  = 256

       self.encoder = nn.Sequential(
           nn.Conv2d(self.l0_chan,  self.l1_chan, 4, stride=2, padding=1),
           nn.BatchNorm2d(self.l1_chan),
           nn.ReLU(),
           nn.Conv2d(self.l1_chan,  self.l2_chan, 4, stride=2, padding=1),
           nn.BatchNorm2d(self.l2_chan),
           nn.ReLU(),
           nn.Conv2d(self.l2_chan, self.l3_chan, 4, stride=2, padding=1),
           nn.BatchNorm2d(self.l3_chan),
           nn.ReLU(),
           nn.Conv2d(self.l3_chan, self.l4_chan, 4, stride=2, padding=1),
           nn.BatchNorm2d(self.l4_chan),
           nn.ReLU(),
           nn.Conv2d(self.l4_chan, self.l5_chan, 4, stride=2, padding=1),
           nn.BatchNorm2d(self.l5_chan),
           nn.ReLU(),

           nn.Flatten(),

           nn.Linear(4*8*self.l5_chan,self.h_neun), #L5
           nn.ReLU(),
           nn.Linear(self.h_neun, self.h_neun),
           nn.ReLU(),
       )

       self.mu  = nn.Linear(self.h_neun, self.latdim)
       self.std = nn.Linear(self.h_neun, self.latdim)

       self.decoder = nn.Sequential(
           nn.Linear(self.latdim,self.h_neun),
           nn.ReLU(),
           nn.Linear(self.h_neun,self.h_neun),
           nn.ReLU(),
           nn.Linear(self.h_neun, 4*8*self.l5_chan), #L5
           nn.ReLU(),

           nn.Unflatten(-1, torch.Size([self.l5_chan, 4, 8])), #L5

           nn.ConvTranspose2d(self.l5_chan, self.l4_chan, 2, stride=2),
           nn.BatchNorm2d(self.l4_chan),
           nn.ReLU(),
           nn.ConvTranspose2d(self.l4_chan, self.l3_chan, 2, stride=2),
           nn.BatchNorm2d(self.l3_chan),
           nn.ReLU(),
           nn.ConvTranspose2d(self.l3_chan, self.l2_chan, 2, stride=2),
           nn.BatchNorm2d(self.l2_chan),
           nn.ReLU(),
           nn.ConvTranspose2d(self.l2_chan, self.l1_chan, 2, stride=2),
           nn.BatchNorm2d(self.l1_chan),
           nn.ReLU(),
           nn.ConvTranspose2d(self.l1_chan, self.l0_chan, 2, stride=2),
           nn.Sigmoid(),
       )

       self.pre = nn.Sequential(
           nn.Conv2d(self.l0_chan,  self.l1_chan, 3, stride=1, padding=1),
           nn.ReLU(),
       )

       self.res = nn.Sequential(
           nn.Conv2d(self.l1_chan,  self.l1_chan, 3, stride=1, padding=1),
           nn.BatchNorm2d(self.l1_chan),
           nn.ReLU(),
           nn.Conv2d(self.l1_chan,  self.l1_chan, 3, stride=1, padding=1),
           nn.BatchNorm2d(self.l1_chan),
           nn.ReLU(),
       )

       self.post = nn.Sequential(
           nn.Conv2d(self.l1_chan,  self.l1_chan, 3, stride=1, padding=1),
           nn.BatchNorm2d(self.l1_chan),
           nn.Dropout2d(p=0.2),
           nn.ReLU(),
           nn.Conv2d(self.l1_chan,  self.l1_chan, 3, stride=1, padding=1),
           nn.BatchNorm2d(self.l1_chan),
           nn.Dropout2d(p=0.2),
           nn.ReLU(),
           nn.Conv2d(self.l1_chan, self.l0_chan, 1, stride=1, padding=0),
           nn.ReLU(),
       )

       self.con = nn.Sequential(
           nn.Conv2d(self.l0_chan, self.l0_chan, 3, stride=1, padding=1),
           nn.Sigmoid(),
       )

   def encode(self, x):
       a    = self.encoder(x)
       mu   = self.mu(a)
       lvar = self.std(a)

       # Reparametrization
       std  = torch.exp(lvar*0.5)
       eps  = torch.randn_like(std)
       z    = mu + eps * std

       return z, mu, lvar

   def decode(self, z):
       return self.decoder(z)

   def forward(self, x):
       z, mu, lvar = self.encode(x)
       y = self.decode(z)
       w = y
       w = self.pre(w)
       for i in range(5):
           d = self.res(w)
           w = w + d
       w = self.post(w)
       y = self.con(w+y)

       return y, mu, lvar


## WORKING 13 #



# ## DISCRIMINATOR 1 ##

# class discrim1(nn.Module):
#    def __init__(self):
#        super().__init__()

#        self.l0_chan = 3
#        self.l1_chan = 6
#        self.l2_chan = 6
#        self.l3_chan = 6
#        self.l4_chan = 9
#        self.l5_chan = 9
#        self.h_neun  = 64

#        self.classifier = nn.Sequential(
#            nn.Conv2d(self.l0_chan,  self.l1_chan, 4, stride=2, padding=1),
#            nn.BatchNorm2d(self.l1_chan),
#            nn.PReLU(),
#            nn.Conv2d(self.l1_chan,  self.l2_chan, 4, stride=2, padding=1),
#            nn.BatchNorm2d(self.l2_chan),
#            nn.PReLU(),
#            nn.Conv2d(self.l2_chan, self.l3_chan, 4, stride=2, padding=1),
#            nn.BatchNorm2d(self.l3_chan),
#            nn.PReLU(),
#            nn.Conv2d(self.l3_chan, self.l4_chan, 4, stride=2, padding=1),
#            nn.BatchNorm2d(self.l4_chan),
#            nn.PReLU(),
#            nn.Conv2d(self.l4_chan, self.l5_chan, 4, stride=2, padding=1),
#            nn.BatchNorm2d(self.l5_chan),
#            nn.PReLU(),

#            nn.Flatten(),

#            nn.Linear(4*8*self.l5_chan,self.h_neun), #L5
#            nn.PReLU(),
#            nn.Linear(self.h_neun, self.h_neun),
#            nn.PReLU(),
#            nn.Linear(self.h_neun, 1),
#            nn.Sigmoid(),
#        )

#    def forward(self, x):
#        y = self.classifier(x)
#        return y

