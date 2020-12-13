import torch
import torch.nn as nn
import torch.nn.functional as F


class InpaintingCNN(nn.Module):
    def __init__(self):
        super(InpaintingCNN, self).__init__()

        # First half - Encoder

        self.c1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.c1_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)

        self.c2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.c2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)

        self.c3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.c3_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)

        self.c4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.c4_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)

        self.c5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.c5_5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        # Decoder
        self.Ct1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, stride=2, kernel_size=4, padding=1)
        self.Ct1_1 = nn.Conv2d(in_channels=512, out_channels=256, stride=1, kernel_size=3, padding=1)
        self.Ct1_c = nn.Conv2d(in_channels=256, out_channels=256, stride=1, kernel_size=3, padding=1)

        self.Ct2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, stride=2, kernel_size=4, padding=1)
        self.Ct2_2 = nn.Conv2d(in_channels=256, out_channels=128, stride=1, kernel_size=3, padding=1)
        self.Ct2_c = nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1)

        self.Ct3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, stride=2, kernel_size=4, padding=1)
        self.Ct3_3 = nn.Conv2d(in_channels=128, out_channels=64, stride=1, kernel_size=3, padding=1)
        self.Ct3_c = nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1)

        self.Ct4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, stride=2, kernel_size=4, padding=1)
        self.Ct4_4 = nn.Conv2d(in_channels=64, out_channels=32, stride=1, kernel_size=3, padding=1)
        self.Ct4_c = nn.Conv2d(in_channels=32, out_channels=32, stride=1, kernel_size=3, padding=1)

        self.final_c = nn.Conv2d(in_channels=32, out_channels=3, stride=1, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = F.relu(self.c1_1(F.relu(self.c1(x))))
        x1_pool = self.pool1(x1)

        x2 = F.relu(self.c2_2(F.relu(self.c2(x1_pool))))
        x2_pool = self.pool2(x2)

        x3 = F.relu(self.c3_3(F.relu(self.c3(x2_pool))))
        x3_pool = self.pool3(x3)

        x4 = F.relu(self.c4_4(F.relu(self.c4(x3_pool))))
        x4_pool = self.pool4(x4)

        encoder_ouput = F.relu(self.c5_5(F.relu(self.c5(x4_pool))))

        # print("ENCODER OP:", encoder_ouput.shape)

        x5 = (F.relu(self.Ct1(encoder_ouput)))
        # print("1ST", x5.shape)
        x5_out = self.Ct1_1(torch.cat((x4, x5), 1))
        # print("AFTER CONCAT",x5_out.shape)
        x5_c = F.relu(self.Ct1_c(x5_out))
        # print("AFTER BLACK",x5_c.shape)

        x6 = (F.relu(self.Ct2(x5_c)))
        # print("1ST", x6.shape)
        x6_out = self.Ct2_2(torch.cat((x3, x6), 1))
        # print("AFTER CONCAT and c ",x6_out.shape) #128,8,8
        x6_c = F.relu(self.Ct2_c(x6_out))
        # print("AFTER BLACK",x6_c.shape)

        x7 = (F.relu(self.Ct3(x6_c)))
        # print("1ST", x7.shape)
        x7_out = self.Ct3_3(torch.cat((x2, x7), 1))
        # print("AFTER CONCAT and c ",x7_out.shape)
        x7_c = F.relu(self.Ct3_c(x7_out))
        # print("AFTER BLACK",x7_c.shape) #64,16,16

        x8 = (F.relu(self.Ct4(x7_c)))
        # print("1ST", x8.shape)
        x8_out = self.Ct4_4(torch.cat((x1, x8), 1))
        # print("AFTER CONCAT and c ",x8_out.shape)
        x8_c = F.relu(self.Ct4_c(x8_out))
        # print("AFTER BLACK",x8_c.shape) #32,32,32

        decoder_output = F.sigmoid(self.final_c(x8_c))
        # print("DECODER OP:",decoder_output.shape)
        return decoder_output
