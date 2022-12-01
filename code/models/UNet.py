import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from .BasicModule import BasicModule
from .ConvLSTM import ConvLSTM
from .ConvRNN import CLSTM_cell
import torch
 
 
 
"""
    构造上采样模块--左边特征提取基础模块    
"""
 
class conv_block(nn.Module):
    """
    Convolution Block
    """
 
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
 
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
 
    def forward(self, x):
        x = self.conv(x)
        return x
 
 
"""
    构造下采样模块--右边特征融合基础模块    
"""
 
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
 
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        x = self.up(x)
        return x
 
"""
    模型主架构
"""
class DNet(BasicModule):
    def __init__(self,conv_dim,layer_num):
        super(DNet,self).__init__()
        layers = []
        layers.append(nn.Conv2d(2, conv_dim,kernel_size=4,stride=2,padding=1))
        layers.append(nn.LeakyReLU(0.2,inplace=True))
        current_dim = conv_dim

        # hidden layers
        for i in range(1,layer_num):
            layers.append(nn.Conv2d(current_dim, current_dim*2, kernel_size=4,stride=2,padding=1))
            layers.append(nn.InstanceNorm2d(current_dim*2))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            current_dim *=2
        self.model = nn.Sequential(*layers)
        self.cl1 = CLSTM_cell(shape=(560,560), input_channels=1, filter_size=1, num_features=1)
        self.conv_src = nn.Conv2d(current_dim,1, kernel_size=3,stride=1,padding=1,bias = False)

    def forward(self,x,c,pattern = 'chanel'):
        if pattern == 'chanel':
            b,t,n,h,w = c.shape
            x = x.reshape(-1,n,h,w)
            c = c.reshape(-1,n,h,w)
            x = self.model(torch.cat([x,c],1))
            out_src = self.conv_src(x)
            _,n,h,w = out_src.shape
            return out_src.reshape(b,t,n,h,w)
        if pattern == 'frame':
            #x = torch.cat([c,x],1)
            b,t,n,h,w = c.shape
            _, hc = self.cl1(c.reshape(t,b,n,h,w))
            x = torch.cat([hc[0],x.squeeze(1)],1)
            #x = x.reshape(b,-1,h,w)
            #x = self.model(torch.cat([x,c],1))
            #x = self.model(hc[0])
            x = self.model(x)
            out_src = self.conv_src(x)
            _,n,h,w = out_src.shape
            return out_src


class UNet(BasicModule):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
 
    # 输入是3个通道的RGB图，输出是0或1——因为我的任务是2分类任务
    def __init__(self, in_ch=20, out_ch=20):
        super(UNet, self).__init__()
 
        # 卷积参数设置
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        # 最大池化层
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
 
        # 左边特征提取卷积层
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.cl1 = ConvLSTM(in_ch, filters[0],[(3,3)],1,True,True,False)
        self.cl2 = ConvLSTM(filters[0], filters[1],[(3,3)],1,True,True,False)
        self.cl3 = ConvLSTM(filters[1], filters[2],[(3,3)],1,True,True,False)
        self.cl4 = ConvLSTM(filters[2], filters[3],[(3,3)],1,True,True,False)
        self.cl5 = ConvLSTM(filters[3], filters[4],[(3,3)],1,True,True,False)
        self.cl1 = CLSTM_cell(shape=(560,560), input_channels=1, filter_size=1, num_features=1)
        self.cl2 = CLSTM_cell(shape=(560,560), input_channels=1, filter_size=1, num_features=1)
        self.cl3 = CLSTM_cell(shape=(560,560), input_channels=6, filter_size=1, num_features=6)
        self.cl4 = CLSTM_cell(shape=(560,560), input_channels=6, filter_size=1, num_features=1)
 
        # 右边特征融合反卷积层
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])
 
        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])
 
        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])
 
        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])
 
        self.Conv = nn.Sequential(nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0),
                                    nn.Tanh())
 
	# 前向计算，输出一张与原图相同尺寸的图片矩阵
    def forward(self, x):
        b,t,c,h,w = x.shape
        cl1,state1 = self.cl1(x.reshape(t,b,c,h,w))
        cl2,state2 = self.cl2(cl1)
        e1 = self.Conv1(cl2.reshape(b,t*c,h,w))
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
 
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
 
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
 
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
 
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)  # 将e4特征图与d5特征图横向拼接
 
        d5 = self.Up_conv5(d5)
 
        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)  # 将e3特征图与d4特征图横向拼接
        d4 = self.Up_conv4(d4)
 
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)  # 将e2特征图与d3特征图横向拼接
        d3 = self.Up_conv3(d3)
 
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)  # 将e1特征图与d1特征图横向拼接
        d2 = self.Up_conv2(d2)
 
        out = self.Conv(d2)
 
        return out
