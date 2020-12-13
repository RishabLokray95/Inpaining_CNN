import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        self.multiChannel = kwargs['multi_channel']
        kwargs.pop('multi_channel')

        self.returnMask = kwargs['return_mask']
        kwargs.pop('return_mask')

        super(PartialConv2d, self).__init__(*args, **kwargs)

        self.weight_mask_updater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                              self.kernel_size[1])

        self.slidingWindowSize = self.weight_mask_updater.shape[1] * self.weight_mask_updater.shape[2] * \
                                 self.weight_mask_updater.shape[3]

        self.size_last = (None, None, None, None)
        self.updateMask = None
        self.ratioOfMask = None

    def forward(self, input, maskIn=None):

        if maskIn is not None or self.size_last != tuple(input.shape):
            self.size_last = tuple(input.shape)

            with torch.no_grad():
                if self.weight_mask_updater.type() != input.type():
                    self.weight_mask_updater = self.weight_mask_updater.to(input)

                if maskIn is None:
                    # if mask is not provided, create a mask
                    if self.multiChannel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = maskIn

                self.updateMask = F.conv2d(mask, self.weight_mask_updater, bias=None, stride=self.stride,
                                           padding=self.padding, dilation=self.dilation, groups=1)

                self.ratioOfMask = self.slidingWindowSize / (self.updateMask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.updateMask = torch.clamp(self.updateMask, 0, 1)
                self.ratioOfMask = torch.mul(self.ratioOfMask, self.updateMask)

        rawOut = super(PartialConv2d, self).forward(torch.mul(input, mask) if maskIn is not None else input)

        if self.bias is not None:
            biasView = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(rawOut - biasView, self.ratioOfMask) + biasView
            output = torch.mul(output, self.updateMask)
        else:
            output = torch.mul(rawOut, self.ratioOfMask)

        if self.returnMask:
            return output, self.updateMask
        else:
            return output
