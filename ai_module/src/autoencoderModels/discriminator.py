import torch.nn as nn

class discriminator(nn.Module):
   def __init__(self):
       super().__init__()

       self.l0_chan = 3
       self.l1_chan = 6
       self.l2_chan = 6
       self.l3_chan = 6
       self.l4_chan = 9
       self.l5_chan = 9
       self.h_neun  = 64

       self.classifier = nn.Sequential(
           nn.Conv2d(self.l0_chan,  self.l1_chan, 4, stride=2, padding=1),
           nn.BatchNorm2d(self.l1_chan),
           nn.PReLU(),
           nn.Conv2d(self.l1_chan,  self.l2_chan, 4, stride=2, padding=1),
           nn.BatchNorm2d(self.l2_chan),
           nn.PReLU(),
           nn.Conv2d(self.l2_chan, self.l3_chan, 4, stride=2, padding=1),
           nn.BatchNorm2d(self.l3_chan),
           nn.PReLU(),
           nn.Conv2d(self.l3_chan, self.l4_chan, 4, stride=2, padding=1),
           nn.BatchNorm2d(self.l4_chan),
           nn.PReLU(),
           nn.Conv2d(self.l4_chan, self.l5_chan, 4, stride=2, padding=1),
           nn.BatchNorm2d(self.l5_chan),
           nn.PReLU(),

           nn.Flatten(),

           nn.Linear(4*8*self.l5_chan,self.h_neun), #L5
           nn.PReLU(),
           nn.Linear(self.h_neun, self.h_neun),
           nn.PReLU(),
           nn.Linear(self.h_neun, 1),
           nn.Sigmoid(),
       )

   def forward(self, x):
       y = self.classifier(x)
       return y
