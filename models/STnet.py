import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class STnet(nn.Module):
    def __init__(self, input_channel, num_classes=3):
        super(STnet, self).__init__()

        # Entry Flow
        self.entry_flow_1 = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=9, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.1),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=9, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.1),
            nn.ReLU(True)
        )

        self.entry_flow_2 = nn.Sequential(
            depthwise_separable_conv(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.1),
            nn.ReLU(True),

            depthwise_separable_conv(128, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.entry_flow_2_residual = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)

        self.entry_flow_3 = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(128, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.Dropout(p=0.1),

            nn.ReLU(True),
            depthwise_separable_conv(256, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.Dropout(p=0.1),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.entry_flow_3_residual = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)

        self.entry_flow_4 = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(256, 728, 3, 1),
            nn.BatchNorm2d(728),
            nn.Dropout(p=0.1),

            nn.ReLU(True),
            depthwise_separable_conv(728, 728, 3, 1),
            nn.BatchNorm2d(728),
            nn.Dropout(p=0.1),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.entry_flow_4_residual = nn.Conv2d(256, 728, kernel_size=1, stride=2, padding=0)

        self.entry_flow_5 = nn.Sequential(
            depthwise_separable_conv(728, 728, 3, 1),
            nn.BatchNorm2d(728),
            nn.Dropout(p=0.1),
            nn.ReLU(True),

            depthwise_separable_conv(728, 728, 3, 1),
            nn.BatchNorm2d(728),
            nn.Dropout(p=0.1),
            nn.ReLU(True)
        )

        self.featurepool_flow_2 = nn.Sequential(
            Flatten(),
            nn.Linear(128*2, 182),
            nn.BatchNorm1d(182),
            nn.Dropout(p=0.1),
            nn.ReLU(True)
        )

        self.featurepool_flow_3 = nn.Sequential(
            Flatten(),
            nn.Linear(256*2, 182),
            nn.BatchNorm1d(182),
            nn.Dropout(p=0.1),
            nn.ReLU(True)
        )

        self.featurepool_flow_45 = nn.Sequential(
            Flatten(),
            nn.Linear(728*2, 182),
            nn.BatchNorm1d(182),
            nn.Dropout(p=0.1),
            nn.ReLU(True)
        )

        # start custom classifier head
        self.poolAvg = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.poolMax = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            Flatten(),
            nn.BatchNorm1d(728),
            nn.Dropout(p=0.3),
            nn.Linear(728, 728),
            #!TODO RELU
            nn.BatchNorm1d(728),
            nn.Dropout(p=0.3),
            nn.Linear(728, num_classes)
        )

    def forward(self, x):
        entry_out1 = self.entry_flow_1(x)
        entry_out2 = self.entry_flow_2(entry_out1) + self.entry_flow_2_residual(entry_out1)
        entry_out3 = self.entry_flow_3(entry_out2) + self.entry_flow_3_residual(entry_out2)
        entry_out4 = self.entry_flow_4(entry_out3) + self.entry_flow_4_residual(entry_out3)
        entry_out5 = self.entry_flow_5(entry_out4)

        pool2 = torch.cat((self.poolAvg(entry_out2), self.poolMax(entry_out2)), dim=1)
        pool3 = torch.cat((self.poolAvg(entry_out3), self.poolMax(entry_out3)), dim=1)
        pool4 = torch.cat((self.poolAvg(entry_out4), self.poolMax(entry_out4)), dim=1)
        pool5 = torch.cat((self.poolAvg(entry_out5), self.poolMax(entry_out5)), dim=1)

        fpool2 = self.featurepool_flow_2(pool2)
        fpool3 = self.featurepool_flow_3(pool3)
        fpool4 = self.featurepool_flow_45(pool4)
        fpool5 = self.featurepool_flow_45(pool5)

        fpool_cat = torch.cat((fpool2, fpool3, fpool4, fpool5), dim=1)

        # pool = torch.cat((self.poolAvg(fpool_cat), self.poolMax(fpool_cat)), dim=1)
        output = self.head(fpool_cat)

        return output

