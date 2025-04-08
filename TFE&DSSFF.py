class TFE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        lms = torch.cat([l, m, s], dim=1)
        return lms

class DSSFF(nn.Module):
    def __init__(self, inc, channel):
        super(DSSFF, self).__init__()
        if channel != inc[0]:
            self.conv0 = Conv(inc[0], channel,1)
        self.conv1 =  Conv(inc[1], channel,1)
        self.conv2 =  Conv(inc[2], channel,1)
        self.conv3d = nn.Conv3d(channel,channel,kernel_size=(1,1,1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3,1,1))
        
        self.dysample1 = DySample(channel, 2, 'lp')
        self.dysample2 = DySample(channel, 4, 'lp')

    def forward(self, x):
        p3, p4, p5 = x[0],x[1],x[2]
        if hasattr(self, 'conv0'):
            p3 = self.conv0(p3)
        p4_2 = self.conv1(p4)
        p4_2 = self.dysample1(p4_2)
        p5_2 = self.conv2(p5)
        p5_2 = self.dysample2(p5_2)
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d],dim = 2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x
      
class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(torch.stack(x, dim=0), dim=0)

