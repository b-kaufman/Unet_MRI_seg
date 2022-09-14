from . import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torch.nn import Dropout
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch

class Block(Module):
    def __init__(self, in_channels, out_channels,dropout_p=.2):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 3)
        self.relu = ReLU()
        self.drop = Dropout(p=dropout_p)
        self.conv2 = Conv2d(out_channels, out_channels, 3)

    def forward(self, x):
        return self.conv2(self.drop(self.relu(self.conv1(x))))

class Encoder(Module):
    def __init__(self, channels=(1,8,16,32,64)):
        super().__init__()
        self.enc_blocks = ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)

    def forward(self, x):

        block_outputs = []


        for block in self.enc_blocks:

            x = block(x)
            block_outputs.append(x)
            x= self.pool(x)

        return block_outputs

class Decoder(Module):
    def __init__(self, channels=(64, 32, 16, 8)):
        super().__init__()

        self.channels = channels
        self.upconvs = ModuleList([ConvTranspose2d(channels[i], channels[i+1],2,2) for i in range(len(channels)-1)])
        self.dec_blocks = ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels) - 1)])
    
    def forward(self, x, enc_features):
        for i in range(len(self.channels)-1):
            x = self.upconvs[i](x)

            enc_feat = self.crop(enc_features[i], x)
            x = torch.cat([x,enc_feat],dim=1)
            x = self.dec_blocks[i](x)

        return x
    def crop(self, enc_features, x):
        (_, _, H, W) = x.shape
        enc_features = CenterCrop([H,W])(enc_features)

        return enc_features

class UNet(Module):
    def __init__(self, enc_channels=(1,8,16,32,64), dec_channels=(64,32,16,8),nbClasses=1, retain_dim=True, out_size=(config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH)):
        super().__init__()
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)
        self.head = Conv2d(dec_channels[-1], nbClasses, 1)
        self.retain_dim = retain_dim
        self.out_size = out_size
    
    def forward(self,x):
        enc_features = self.encoder(x)

        dec_features = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])

        smap = self.head(dec_features)

        if self.retain_dim:
            smap = F.interpolate(smap, self.out_size)

        return smap

if __name__ == "__main__":
    UNet()
