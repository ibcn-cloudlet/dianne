#!/usr/bin/env python

import torch
import torch.onnx
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy


def export(model, dummy_input):
    print "Model"
    print model
    print

    print "Input (see input.csv)"
    inp = dummy_input.data.numpy()
    print inp
    if len(inp.shape) == 4:
        numpy.savetxt("input.csv", dummy_input.data.numpy()[0, 0], delimiter=",")
    else:
        numpy.savetxt("input.csv", dummy_input.data.numpy(), delimiter=",")
    print

    print "Output (see output.csv)"
    outp = model(dummy_input).data.numpy()
    print outp
    if len(outp.shape) == 4:
        numpy.savetxt("output.csv", outp[0, 0], delimiter=",")
    else:
        numpy.savetxt("output.csv", outp, delimiter=",")

    print
    torch.onnx.export(model, dummy_input, "model.pb", verbose=True)


def singleFF():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(in_features=32, out_features=5)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            return x

    model = Net()
    dummy_input = Variable(torch.randn(1, 32))
    return model, dummy_input


def singleConv():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, dilation=1,
                                   groups=1, bias=True)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            return x

    model = Net()
    dummy_input = Variable(torch.randn(1, 1, 10, 10))
    return model, dummy_input


def convLinear():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, dilation=1,
                                   groups=1, bias=True)
            self.fc1 = nn.Linear(in_features=64, out_features=5)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = x.view(-1, 64)
            x = self.fc1(x)
            x = F.relu(x)
            return x

    model = Net()
    dummy_input = Variable(torch.randn(1, 1, 10, 10))
    return model, dummy_input


def convLinearPool():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, dilation=1,
                                   groups=1, bias=True)
            self.fc1 = nn.Linear(in_features=49, out_features=5)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = F.max_pool2d(input=x, kernel_size=2, stride=1)
            x = x.view(-1, 49)
            x = self.fc1(x)
            x = F.relu(x)
            return x

    model = Net()
    dummy_input = Variable(torch.randn(1, 1, 10, 10))
    return model, dummy_input



if __name__ == "__main__":
    model, dummy_input = singleFF()
    #model, dummy_input = singleConv()
    #model, dummy_input = convLinear()
    #model, dummy_input = convLinearPool()
    export(model, dummy_input)
