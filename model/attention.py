import torch 
import torch.nn as nn 

# structure of module based off of this source
# https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html

# more information about attention from the following sources
# https://towardsdatascience.com/attention-and-its-different-forms-7fc3674d14dc
# https://medium.com/mlearning-ai/self-attention-in-convolutional-neural-networks-172d947afc00
# https://iopscience.iop.org/article/10.1088/1742-6596/1693/1/012173/pdf

class SelfAttention(nn.Module):

    def __init__(self, n_channels):
        
        # create 3 convolutional layers f(x), g(x), h(x)
        self.query, self.key, self.value = [self._conv(n_channels, c) for c in (n_channels//8, n_channels//8, n_channels)]
        self.gamma = nn.Parameter(tensor([0.]))

    def _conv(self, n_in, n_out):

        # use spectral normalization since it only has one hyperparameter to tune
        return ConvLayer(n_in, n_out, ks=1, ndim=1, norm_type=NormType.Spectral, act_cls=None, bias=False)

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2], -1)
        
        f = self.query(x)
        g = self.key(x)
        h = self.value(x)

        # find the softmax attention weights
        # torch.bmm() achieves batch matrix multiplication 
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x 

        return o.view(*size).contiguous()