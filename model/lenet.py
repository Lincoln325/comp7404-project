import torch.nn as nn
from collections import OrderedDict

from torch.nn import init


class C1(nn.Module):
    def __init__(self, dilation):
        super(C1, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 6, kernel_size=(3, 3), padding=0, dilation=dilation)),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output


class C2(nn.Module):
    def __init__(self, dilation):
        super(C2, self).__init__()

        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(6, 12, kernel_size=(3, 3), padding=0, dilation=dilation)),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c2(img)
        return output


class C3(nn.Module):
    def __init__(self, dilation):
        super(C3, self).__init__()

        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(12, 32, kernel_size=(3, 3), padding=0, dilation=dilation)),
            ('relu3', nn.ReLU()),
            ('s3', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c3(img)
        return output
    
class C4(nn.Module):
    def __init__(self, dilation):
        super(C4, self).__init__()

        self.c4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv2d(32, 64, kernel_size=(3, 3), padding="same", dilation=dilation)),
            ('relu4', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c4(img)
        return output

class C5(nn.Module):
    def __init__(self, dilation):
        super(C5, self).__init__()

        self.c5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv2d(64, 96, kernel_size=(3, 3), padding="same", dilation=dilation)),
            ('relu5', nn.ReLU()),
            ('s5', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c5(img)
        return output
    
class C6(nn.Module):
    def __init__(self, dilation):
        super(C6, self).__init__()

        self.c6 = nn.Sequential(OrderedDict([
            ('c6', nn.Conv2d(96, 128, kernel_size=(3, 3), padding="same", dilation=dilation)),
            ('relu6', nn.ReLU()),
            ('s6', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c6(img)
        return output


class F7(nn.Module):
    def __init__(self):
        super(F7, self).__init__()

        self.f7 = nn.Sequential(OrderedDict([
            ('f7', nn.Linear(512, 128)),
            # ('relu7', nn.ReLU()),
            # ('dropout7', nn.Dropout(p=0.2))
        ]))

    def forward(self, img):
        output = self.f7(img)
        return output


class F8(nn.Module):
    def __init__(self, class_num):
        super(F8, self).__init__()

        self.f8 = nn.Sequential(OrderedDict([
            ('f8', nn.Linear(128, class_num)),
            # ('relu8', nn.ReLU()),
            # ('sig8', nn.LogSoftmax(dim=-1)),
            # ('dropout8', nn.Dropout(p=0.2))
        ]))

    def forward(self, img):
        output = self.f8(img)
        return output


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self, class_num, dilation):
        super(LeNet5, self).__init__()

        self.c1 = C1(dilation[0])
        self.c2 = C2(dilation[1])
        self.c3 = C3(dilation[2])
        self.c4 = C4(dilation[3])
        self.c5 = C5(dilation[4])
        self.c6 = C6(dilation[5])
        self.pooling = nn.AdaptiveMaxPool1d(512)
        self.f7 = F7()
        self.f8 = F8(class_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, img):
        output = self.c1(img)
        output = self.c2(output)
        output = self.c3(output)
        output = self.c4(output)
        output = self.c5(output)
        output = self.c6(output)

        # x = self.c2_1(output)
        # output = self.c2_2(output)

        # output += x

        # output = self.c3(output)
        output = output.view(img.size(0), -1)
        output = self.pooling(output)
        output = self.f7(output)
        output = self.f8(output)
        return output