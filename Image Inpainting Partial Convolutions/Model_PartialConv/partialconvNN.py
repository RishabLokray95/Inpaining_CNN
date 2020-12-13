import torch
import torch.nn.functional as F
from torch import nn
from Model_PartialConv.partialconv_decoder import PartialConvolutionDecoder
from Model_PartialConv.partialconv_encoder import PartialConvolutionEncoder


class ImageinpaintingPConvNN(nn.Module):
    def __init__(self, channels=3):
        super(ImageinpaintingPConvNN, self).__init__()
        self.encoder1 = PartialConvolutionEncoder(channels, 64, 7, bn=False)
        self.encoder2 = PartialConvolutionEncoder(64, 128, 5)
        self.encoder3 = PartialConvolutionEncoder(128, 256, 5)
        self.encoder4 = PartialConvolutionEncoder(256, 512, 3)
        self.encoder5 = PartialConvolutionEncoder(512, 512, 3)

        self.decoder1 = PartialConvolutionDecoder(512 + 512, 512, 3)
        self.decoder2 = PartialConvolutionDecoder(512 + 256, 256, 3)
        self.decoder3 = PartialConvolutionDecoder(256 + 128, 128, 3)
        self.decoder4 = PartialConvolutionDecoder(128 + 64, 64, 3)
        self.decoder5 = PartialConvolutionDecoder(64 + 3, 3, 3, bn=False, act=True, return_mask=False)
        self.convfinal = nn.Conv2d(3, 3, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        inputs_img, inputs_mask = x
        # print('Input')
        # print('Image Size : '+str(inputs_img.size()))
        # print('Mask Size : '+str(inputs_mask.size()))
        e_conv1, e_mask1 = self.encoder1(inputs_img, inputs_mask)
        # print('Encoder Layer 1')
        # print('Image Size : '+str(e_conv1.size()))
        # print('Mask Size : '+str(e_mask1.size()))
        e_conv2, e_mask2 = self.encoder2(e_conv1, e_mask1)
        # print('Encoder Layer 2')
        # print('Image Size : ' + str(e_conv2.size()))
        # print('Mask Size : ' + str(e_mask2.size()))
        e_conv3, e_mask3 = self.encoder3(e_conv2, e_mask2)
        # print('Encoder Layer 3')
        # print('Image Size : ' + str(e_conv3.size()))
        # print('Mask Size : ' + str(e_mask3.size()))
        e_conv4, e_mask4 = self.encoder4(e_conv3, e_mask3)
        # print('Encoder Layer 4')
        # print('Image Size : ' + str(e_conv4.size()))
        # print('Mask Size : ' + str(e_mask4.size()))
        e_conv5, e_mask5 = self.encoder5(e_conv4, e_mask4)
        # print('Encoder Layer 5')
        # print('Image Size : ' + str(e_conv5.size()))
        # print('Mask Size : ' + str(e_mask5.size()))

        d_conv6, d_mask6 = self.decoder1(e_conv5, e_mask5, e_conv4, e_mask4)
        # print('Decoder Layer 1')
        # print('Image Size : ' + str(d_conv6.size()))
        # print('Mask Size : ' + str(d_mask6.size()))
        d_conv7, d_mask7 = self.decoder2(d_conv6, d_mask6, e_conv3, e_mask3)
        # print('Decoder Layer 2')
        # print('Image Size : ' + str(d_conv7.size()))
        # print('Mask Size : ' + str(d_mask7.size()))
        d_conv8, d_mask8 = self.decoder3(d_conv7, d_mask7, e_conv2, e_mask2)
        # print('Decoder Layer 3')
        # print('Image Size : ' + str(d_conv8.size()))
        # print('Mask Size : ' + str(d_mask8.size()))
        d_conv9, d_mask9 = self.decoder4(d_conv8, d_mask8, e_conv1, e_mask1)
        # print('Decoder Layer 4')
        # print('Image Size : ' + str(d_conv9.size()))
        # print('Mask Size : ' + str(d_mask9.size()))
        output = self.decoder5(d_conv9, d_mask9, inputs_img, inputs_mask)
        self.convfinal(output)
        output = self.sigmoid(output)

        return output
