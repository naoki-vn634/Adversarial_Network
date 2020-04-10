import torch.nn as nn


class DCGenerator(nn.Module):
    def __init__(self,num_chnannel=1,input_dim=20,image_size=64):
        #inputは乱数その次元がinput_dim(例:torch.randn(1,20,1,1))
        super(DCGenerator,self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(input_dim,image_size*8,kernel_size=4,stride=1),
            nn.BatchNorm2d(image_size*8),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(image_size*8,image_size*4,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(image_size*4),
            nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(image_size*4,image_size,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True))

        # self.layer4 = nn.Sequential(
        #     nn.ConvTranspose2d(image_size*4,image_size*2,kernel_size=4,stride=2,padding=1),
        #     nn.BatchNorm2d(image_size*2),
        #     nn.ReLU(inplace=True))

        # self.layer5 = nn.Sequential(
        #     nn.ConvTranspose2d(image_size*2,image_size,kernel_size=4,stride=2,padding=1),
            # nn.BatchNorm2d(image_size),
            # nn.ReLU(inplace=True))
        
        self.last = nn.Sequential(
            nn.ConvTranspose2d(image_size,num_chnannel,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self,x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        # h = self.layer4(h)
        # h = self.layer5(h)
        out = self.last(h)

        return out

class DCDiscriminator(nn.Module):
    def __init__(self,num_channel=1, image_size=64):
        super(DCDiscriminator,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channel,image_size,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(image_size,image_size*4,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer3 = nn.Sequential(
            nn.Conv2d(image_size*4,image_size*8,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.1, inplace=True))

        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(image_size*4,image_size*8,kernel_size=4,stride=2,padding=1),
        #     nn.LeakyReLU(0.1, inplace=True))

        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(image_size*8,image_size*16,kernel_size=4,stride=2,padding=1),
        #     nn.LeakyReLU(0.1, inplace=True))

        self.last = nn.Conv2d(image_size*8,1,kernel_size=4,stride=1)

    def forward(self,x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        # h = self.layer4(h)
        # h = self.layer5(h)
        out = self.last(h)

        return out


