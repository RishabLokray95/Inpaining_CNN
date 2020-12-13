import torch
import torch.nn.functional as F
from torch import nn
from Model_PartialConv.partial_conv2d import PartialConv2d


class PartialConvolutionDecoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, bn=True, act=True, return_mask=True):
        super(PartialConvolutionDecoder, self).__init__()
        self.upsample_img = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample_mask = nn.UpsamplingNearest2d(scale_factor=2)

        self.pconv = PartialConv2d(input_channels, output_channels, kernel_size=kernel_size, stride=1,
                                   padding=(kernel_size - 1) // 2, multi_channel=True, return_mask=return_mask)
        self.bn = bn
        self.batchnorm = nn.BatchNorm2d(output_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.act = act
        self.return_mask = return_mask

    def forward(self, img, mask_in, e_conv, e_mask):
        upsampledImage = self.upsample_img(img)
        upsampledMask = self.upsample_mask(mask_in)
        concatinatedImage = torch.cat([e_conv, upsampledImage], dim=1)
        concatinatedMask = torch.cat([e_mask, upsampledMask], dim=1)
        if self.return_mask:
            convolutedOut, mask = self.pconv(concatinatedImage, concatinatedMask)
        else:
            convolutedOut = self.pconv(concatinatedImage, concatinatedMask)
        if self.bn:
            convolutedOut = self.batchnorm(convolutedOut)
        if self.act:
            convolutedOut = self.activation(convolutedOut)

        if self.return_mask:
            return convolutedOut, mask
        else:
            return convolutedOut