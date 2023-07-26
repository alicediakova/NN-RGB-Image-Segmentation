import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniUNet(nn.Module):
    # TODO: implement a neural network as described in the handout
    def __init__(self):
        """Initialize the layers of the network as instance variables."""
        super(MiniUNet, self).__init__()
        # TODO
        
        #convolution operations:
        self.c1 = nn.Conv2d(3, 16, kernel_size = 3, padding = 1)
        self.c2 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
        self.c3 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.c4 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        
        self.c5 = nn.Conv2d(128, 256, kernel_size = 3, padding = 1)
        
        self.c6 = nn.Conv2d(128+256, 128, kernel_size = 3, padding = 1)
        self.c7 = nn.Conv2d(64+128, 64, kernel_size = 3, padding = 1)
        self.c8 = nn.Conv2d(32+64, 32, kernel_size = 3, padding = 1)
        self.c9 = nn.Conv2d(16+32, 16, kernel_size = 3, padding = 1)
        
        self.c10 = nn.Conv2d(16, 6, kernel_size = 1, padding = 0)
        
        # max pooling operation:
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """
        In:
            x: Tensor [batchsize, channel, height, width], channel=3 for rgb input
        Out:
            output: Tensor [batchsize, class, height, width], class=number of objects + 1 for background
        Purpose:
            Forward process. Pass the input x through the layers defined in __init__() to get the output.
        """
        # TODO
        output = None
        a = F.relu(self.c1(x)) # note: concat before c9
        #print("a: " + str(a.size()))
        b = F.relu(self.c2(self.pool(a)))
        #print("b: " + str(b.size()))
        c = F.relu(self.c3(self.pool(b)))
        #print("c: " + str(c.size()))
        d = F.relu(self.c4(self.pool(c)))
        #print("d: " + str(d.size()))
        e = F.relu(self.c5(self.pool(d)))
        #print("e: " + str(e.size()))
        
        e = F.interpolate(e, scale_factor = 2)
        #print("e1: " + str(e.size()))
        e = torch.cat((d, e), 1)
        #print("e2: " + str(e.size()))
        e = F.relu(self.c6(e))
        #print("e3: " + str(e.size()))
        
        e = F.interpolate(e, scale_factor = 2)
        #print("e1: " + str(e.size()))
        e = torch.cat((c, e), 1)
        #print("e2: " + str(e.size()))
        e = F.relu(self.c7(e))
        #print("e3: " + str(e.size()))
        
        e = F.interpolate(e, scale_factor = 2)
        #print("e1: " + str(e.size()))
        e = torch.cat((b, e), 1)
        #print("e2: " + str(e.size()))
        e = F.relu(self.c8(e))
        #print("e3: " + str(e.size()))
        
        e = F.interpolate(e, scale_factor = 2)
        #print("e1: " + str(e.size()))
        e = torch.cat((a, e), 1)
        #print("e2: " + str(e.size()))
        e = F.relu(self.c9(e))
        #print("e3: " + str(e.size()))
        
        output = self.c10(e)
        return output


if __name__ == '__main__':
    model = MiniUNet()
    input_tensor = torch.zeros([1, 3, 240, 320])
    output = model(input_tensor)
    print("output size:", output.size())
    print(model)
