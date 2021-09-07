import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet_brain(nn.Module):
    def consecutive_conv(self, in_channels, out_channels, mid_channels = 0):
        if mid_channels == 0:
            mid_channels = out_channels
        return nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True))
    
    def consecutive_conv_f(self, in_channels, out_channels, mid_channels = 0):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.Sigmoid())

    def __init__(self, num_channels=64):
        super(Unet_brain, self).__init__()

        self.conv_initial = self.consecutive_conv(1, num_channels)

        self.conv_rest1 = self.consecutive_conv(num_channels, num_channels * 2)
        self.conv_rest2 = self.consecutive_conv(num_channels * 2, num_channels * 4)
        self.conv_rest3 = self.consecutive_conv(num_channels * 4, num_channels * 8)
        self.conv_bottom = self.consecutive_conv(num_channels * 8, num_channels * 8, num_channels * 16)
        
        self.conv_cls1 = self.consecutive_conv(num_channels * 8, num_channels * 4)
        self.conv_cls2 = self.consecutive_conv(num_channels * 4, num_channels * 2)
        self.dense = nn.Linear(num_channels*2, 1)
        #self.dense_ = nn.Linear(num_channels*2, 3)

        self.conv_up1 = self.consecutive_conv(num_channels * 16, num_channels * 8)
        self.conv_up2 = self.consecutive_conv(num_channels * 8, num_channels * 4)
        self.conv_up3 = self.consecutive_conv(num_channels * 4, num_channels * 2)
        self.conv_up4 = self.consecutive_conv(num_channels * 2, num_channels)

        self.conv_final = self.consecutive_conv_f(num_channels+1, 1)

        self.pooling = nn.MaxPool3d(2, stride=2)

        self.upsample1 = nn.ConvTranspose3d(num_channels*8, num_channels*8, 2, 2)
        self.upsample2 = nn.ConvTranspose3d(num_channels*8, num_channels*4, 2, 2)
        self.upsample3 = nn.ConvTranspose3d(num_channels*4, num_channels*2, 2, 2)
        self.upsample4 = nn.ConvTranspose3d(num_channels*2, num_channels*1, 2, 2)

    def forward(self, x):
        # 2*64*64*64 to 64*32*32*32
        x_64 = self.conv_initial(x)
        x_32 = self.pooling(x_64)

        # 64*32*32*32 to 128*16*16*16
        x_32 = self.conv_rest1(x_32)
        x_16 = self.pooling(x_32)

        # 128*16*16*16 to 256*8*8*8
        x_16 = self.conv_rest2(x_16)
        x_8 = self.pooling(x_16)

        # 256*8*8*8 to 512*4*4*4
        x_8 = self.conv_rest3(x_8)
        x_4 = self.pooling(x_8)

        # 64*16*16*16 to 64*8*8*8
        x_4 = self.conv_bottom(x_4)      

        #classification path
        c_4 = self.conv_cls1(x_4)   
        c_2 = self.pooling(c_4)

        c_2 = self.conv_cls2(c_2)    
        c_1 = self.pooling(c_2)
    
        c_1 = self.dense(torch.flatten(c_1, start_dim=1))
        
        cls = torch.sigmoid(c_1)

        # upsmapling path
        u_8 = self.upsample1(x_4)
        u_8 = self.conv_up1(torch.cat((x_8, u_8), 1))

        u_16 = self.upsample2(u_8)
        u_16 = self.conv_up2(torch.cat((x_16, u_16), 1))

        u_32 = self.upsample3(u_16)
        u_32 = self.conv_up3(torch.cat((x_32, u_32), 1))

        u_64 = self.upsample4(u_32)
        u_64 = self.conv_up4(torch.cat((x_64, u_64), 1))

        cls_to_dec = cls.view(cls.shape[0],1,1,1,cls.shape[1])
        cls_to_dec = cls_to_dec.repeat(1,1,u_64.shape[2],u_64.shape[3],u_64.shape[4])
        u_64 = torch.cat((u_64, cls_to_dec),1)

        seg = self.conv_final(u_64)   
  
        return seg, cls

    
class Unet_abdom(nn.Module):
    def consecutive_conv(self, in_channels, out_channels, mid_channels = 0):
        if mid_channels == 0:
            mid_channels = out_channels
        return nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True))
    
    def consecutive_conv_f(self, in_channels, out_channels, mid_channels = 0):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.Sigmoid())

    def __init__(self, num_channels=64):
        super(Unet_abdom, self).__init__()

        self.conv_initial = self.consecutive_conv(28, num_channels)

        self.conv_rest1 = self.consecutive_conv(num_channels, num_channels * 2)
        self.conv_rest2 = self.consecutive_conv(num_channels * 2, num_channels * 4)
        self.conv_rest3 = self.consecutive_conv(num_channels * 4, num_channels * 8)
        self.conv_bottom = self.consecutive_conv(num_channels * 8, num_channels * 8, num_channels * 16)
        
        self.conv_cls1 = self.consecutive_conv(num_channels * 8, num_channels * 4)
        self.conv_cls2 = self.consecutive_conv(num_channels * 4, num_channels * 2)
        self.conv_cls3 = self.consecutive_conv(num_channels * 2, num_channels * 1)
        self.dense = nn.Linear(num_channels, 1)

        self.conv_up1 = self.consecutive_conv(num_channels * 16, num_channels * 8)
        self.conv_up2 = self.consecutive_conv(num_channels * 8, num_channels * 4)
        self.conv_up3 = self.consecutive_conv(num_channels * 4, num_channels * 2)
        self.conv_up4 = self.consecutive_conv(num_channels * 2, num_channels)

        self.conv_final = self.consecutive_conv_f(num_channels, 1)

        self.pooling = nn.MaxPool3d(2, stride=2)

        self.upsample1 = nn.ConvTranspose3d(num_channels*8, num_channels*8, 2, 2)
        self.upsample2 = nn.ConvTranspose3d(num_channels*8, num_channels*4, 2, 2)
        self.upsample3 = nn.ConvTranspose3d(num_channels*4, num_channels*2, 2, 2)
        self.upsample4 = nn.ConvTranspose3d(num_channels*2, num_channels*1, 2, 2)

    def forward(self, x):
        # 2*64*64*64 to 64*32*32*32
        x_64 = self.conv_initial(x)
        x_32 = self.pooling(x_64)

        # 64*32*32*32 to 128*16*16*16
        x_32 = self.conv_rest1(x_32)
        x_16 = self.pooling(x_32)

        # 128*16*16*16 to 256*8*8*8
        x_16 = self.conv_rest2(x_16)
        x_8 = self.pooling(x_16)

        # 256*8*8*8 to 512*4*4*4
        x_8 = self.conv_rest3(x_8)
        x_4 = self.pooling(x_8)

        # 64*16*16*16 to 64*8*8*8
        x_4 = self.conv_bottom(x_4)      

        # upsmapling path
        u_8 = self.upsample1(x_4)
        u_8 = self.conv_up1(torch.cat((x_8, u_8), 1))

        u_16 = self.upsample2(u_8)
        u_16 = self.conv_up2(torch.cat((x_16, u_16), 1))

        u_32 = self.upsample3(u_16)
        u_32 = self.conv_up3(torch.cat((x_32, u_32), 1))

        u_64 = self.upsample4(u_32)
        u_64 = self.conv_up4(torch.cat((x_64, u_64), 1))

        seg = self.conv_final(u_64)   
        
        #classification path
        c_4 = self.conv_cls1(x_4)   
        c_2 = self.pooling(c_4)

        c_2 = self.conv_cls2(c_2)    
        c_1_ = self.pooling(c_2)
        
        c_1_ = self.conv_cls3(c_1_)  
        c_1 = self.pooling(c_1_)
    
        c_1 = self.dense(torch.flatten(c_1, start_dim=1))
        
        cls = torch.sigmoid(c_1)

        return seg, cls
