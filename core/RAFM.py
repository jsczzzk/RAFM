import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass





class SepConvGRU(nn.Module):
    def __init__(self):
        super(SepConvGRU, self).__init__()
        hidden_dim = 128
        catt = 256

        self.convz1 = nn.Conv2d(catt, hidden_dim, (1, 3), padding=(0, 1))
        self.convr1 = nn.Conv2d(catt, hidden_dim, (1, 3), padding=(0, 1))
        self.convq1 = nn.Conv2d(catt, hidden_dim, (1, 3), padding=(0, 1))

        self.convz2 = nn.Conv2d(catt, hidden_dim, (3, 1), padding=(1, 0))
        self.convr2 = nn.Conv2d(catt, hidden_dim, (3, 1), padding=(1, 0))
        self.convq2 = nn.Conv2d(catt, hidden_dim, (3, 1), padding=(1, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class RAFM(nn.Module):
    def __init__(self):
        super(RAFM, self).__init__()

        self.convX31 = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1,dilation=1,bias=True),
        torch.nn.Tanh())

        self.convX32= torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=2,dilation=2,bias=True),
        torch.nn.Tanh())

        self.convX33 = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=4,dilation=4,bias=True),
        torch.nn.Tanh())

        self.convX34= torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=6,dilation=6,bias=True),
        torch.nn.Tanh())

        self.convX35 = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=8,dilation=8,bias=True),
        torch.nn.Tanh())

        self.convX36 = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=12,dilation=12,bias=True),
        torch.nn.Tanh())


        self.sigmoid = nn.Sigmoid()

        self.update_block = BasicUpdateBlock()
        self.gruc = SepConvGRU()
    def upsample_depth(self, flow, mask):
        """ Upsample depth field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 1, 8 * H, 8 * W)

    def forward(self, features, iters=6):
        """ Estimate depth for a single image """

        x1, x2, x3 = features
        x31, x32, x33, x34, x35,x36 = self.convX31(x3), self.convX32(x3), self.convX33(x3), self.convX34(x3), self.convX35(x3), self.convX36(x3)
        disp_predictions = {}
        b, c, h, w = x3.shape
        dispFea = torch.zeros([b, 1, h, w], requires_grad=True).to(x1.device)
        net = torch.zeros([b, 256, h, w], requires_grad=True).to(x1.device)

        for itr in range(iters):
            if itr in [0]:
                corr = x31
            elif itr in [1]:
                corrh = corr
                corr = x32 
                corr = self.gruc(corrh, corr)
            elif itr in [2]:
                corrh = corr
                corr = x33 
                corr = self.gruc(corrh, corr)
            elif itr in [3]:
                corrh = corr
                corr = x34 
                corr = self.gruc(corrh, corr)   
            elif itr in [4]:
                corrh = corr
                corr = x35 
                corr = self.gruc(corrh, corr)   
            elif itr in [5]:
                corrh = corr
                corr = x36 
                corr = self.gruc(corrh, corr)   

            net, up_mask, delta_disp = self.update_block(net, corr, dispFea)
            dispFea = dispFea + delta_disp

            disp = self.sigmoid(dispFea)
            # upsample predictions
            if self.training:
                disp_up = self.upsample_depth(disp, up_mask)
                disp_predictions[("disp_up", itr)] = disp_up
            else:
                if (iters-1)==itr:
                    disp_up = self.upsample_depth(disp, up_mask)
                    disp_predictions[("disp_up", itr)] = disp_up


        return disp_predictions

