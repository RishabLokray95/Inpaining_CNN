from torch import nn
from Model_PartialConv.partial_conv2d import PartialConv2d


class PartialConvolutionEncoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, bn=True):
        super(PartialConvolutionEncoder, self).__init__()
        self.pconv = PartialConv2d(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                                   padding=(kernel_size - 1) // 2, multi_channel=True, return_mask=True)
        self.bn = bn
        self.batchnormalization = nn.BatchNorm2d(output_channels)
        self.activationfunction = nn.ReLU()

    def forward(self, img, mask_in):
        conv, mask = self.pconv(img, mask_in)
        if self.bn:
            conv = self.batchnormalization(conv)
        conv = self.activationfunction(conv)

        return conv, mask