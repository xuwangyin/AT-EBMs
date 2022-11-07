'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class FakeReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SequentialWithArgs(torch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        vs = list(self._modules.values())
        l = len(vs)
        for i in range(l):
            if i == l - 1:
                input = vs[i](input, *args, **kwargs)
            else:
                input = vs[i](input)
        return input


class DownscaleBasicBlockBN(nn.Module):
    def __init__(self, in_planes, tmp_planes, out_planes):
        super(DownscaleBasicBlockBN, self).__init__()
        self.in_planes = in_planes
        if in_planes == 3:
            self.fromrgb = nn.Conv2d(in_planes, tmp_planes, kernel_size=1, stride=1, padding=0, bias=True)

        self.conv1 = nn.Conv2d(tmp_planes, tmp_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(tmp_planes)
        self.conv2 = nn.Conv2d(tmp_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes)

        shortcut_input_planes = in_planes if in_planes != 3 else tmp_planes
        self.shortcut = nn.Conv2d(shortcut_input_planes, out_planes, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        if self.in_planes == 3:
            x = F.leaky_relu(self.fromrgb(x))
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out += F.leaky_relu(self.bn3(self.shortcut(x)))
        return out


class ResNet6BNStyleGAN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
         
        self.num_classes = num_classes
        self.B0 = DownscaleBasicBlockBN(3, 64, 128) # Outputs 128x128x128
        self.B1 = DownscaleBasicBlockBN(128, 128, 256) # Outputs 256x64x64 
        self.B2 = DownscaleBasicBlockBN(256, 256, 512) # Outputs 512x32x32
        self.B3 = DownscaleBasicBlockBN(512, 512, 512) # Outputs 512x16x16
        self.B4 = DownscaleBasicBlockBN(512, 512, 512) # Outputs 512x8x8
        self.B5 = DownscaleBasicBlockBN(512, 512, 512) # Outputs 512x4x4
        self.conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True) # Outputs 512x4x4
        self.fc = nn.Linear(512*4*4, 512)
        self.out = nn.Linear(512, self.num_classes)
        

    def forward(self, x):
        out = self.B5(self.B4(self.B3(self.B2(self.B1(self.B0(x))))))
        out = F.leaky_relu(self.conv(out))
        out = F.leaky_relu(self.fc(out.flatten(1)))
        out = self.out(out)
        return out


class ResNet5BN(nn.Module):
    def __init__(self, num_classes):
        super(ResNet5BN, self).__init__()

        self.B0 = DownscaleBasicBlockBN(3, 64)
        self.B1 = DownscaleBasicBlockBN(64, 128)
        self.B2 = DownscaleBasicBlockBN(128, 256)
        self.B3 = DownscaleBasicBlockBN(256, 256)
        self.B4 = DownscaleBasicBlockBN(256, 512)
        self.B5 = DownscaleBasicBlockBN(512, 512)
        self.pool = nn.AvgPool2d(2)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.B5(self.B4(self.B3(self.B2(self.B1(self.B0(x))))))
        out = self.pool(out)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        return final

class ResNet6BN(nn.Module):
    def __init__(self, num_classes):
        super(ResNet6BN, self).__init__()

        self.B0 = DownscaleBasicBlockBN(3, 64)
        self.B1 = DownscaleBasicBlockBN(64, 128)
        self.B2 = DownscaleBasicBlockBN(128, 256)
        self.B3 = DownscaleBasicBlockBN(256, 256)
        self.B4 = DownscaleBasicBlockBN(256, 512)
        self.B5 = DownscaleBasicBlockBN(512, 512)
        self.B6 = DownscaleBasicBlockBN(512, 512)
        self.pool = nn.AvgPool2d(2)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.B6(self.B5(self.B4(self.B3(self.B2(self.B1(self.B0(x)))))))
        out = self.pool(out)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        return final


class ResNet6BNWide(nn.Module):
    def __init__(self, num_classes):
        super(ResNet6BNWide, self).__init__()

        self.B0 = DownscaleBasicBlockBN(3, 64)
        self.B1 = DownscaleBasicBlockBN(64, 128)
        self.B2 = DownscaleBasicBlockBN(128, 256)
        self.B3 = DownscaleBasicBlockBN(256, 256)
        self.B4 = DownscaleBasicBlockBN(256, 512)
        self.B5 = DownscaleBasicBlockBN(512, 512)
        self.B6 = DownscaleBasicBlockBN(512, 1024)
        self.pool = nn.AvgPool2d(2)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.B6(self.B5(self.B4(self.B3(self.B2(self.B1(self.B0(x)))))))
        out = self.pool(out)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        return final


class DownscaleBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes):
        super(DownscaleBasicBlock, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_planes, planes, kernel_size=3, stride=1,
                                             padding=1, bias=True))
        self.conv2 = spectral_norm(nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                             padding=1, bias=True))
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.shortcut = spectral_norm(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=3,
                                                padding=1, stride=1, bias=True))

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool(self.conv2(out))
        out += self.pool(self.shortcut(x))
        return F.relu(out)


class ResNet5(nn.Module):
    def __init__(self, num_classes):
        super(ResNet5, self).__init__()

        self.B0 = DownscaleBasicBlock(3, 64)
        self.B1 = DownscaleBasicBlock(64, 128)
        self.B2 = DownscaleBasicBlock(128, 256)
        self.B3 = DownscaleBasicBlock(256, 256)
        self.B4 = DownscaleBasicBlock(256, 512)
        self.B5 = DownscaleBasicBlock(512, 512)
        self.pool = nn.AvgPool2d(2)
        self.linear = spectral_norm(nn.Linear(512, num_classes))

    def forward(self, x):
        out = self.B5(self.B4(self.B3(self.B2(self.B1(self.B0(x))))))
        out = self.pool(out)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        return final


class ResNet3(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3, self).__init__()

        self.B0 = DownscaleBasicBlock(3, 128)
        self.B1 = DownscaleBasicBlock(128, 128)
        self.B2 = BasicBlock(128, 128)
        self.B3 = BasicBlock(128, 128)
        self.pool = nn.AvgPool2d(8)
        self.linear = spectral_norm(nn.Linear(128, num_classes))

    def forward(self, x):
        out = self.B3(self.B2(self.B1(self.B0(x))))
        out = self.pool(out)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        return final


class ResNet3BN(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3BN, self).__init__()

        self.B0 = DownscaleBasicBlockBN(3, 128)
        self.B1 = DownscaleBasicBlockBN(128, 128)
        self.B2 = BasicBlock(128, 128)
        self.B3 = BasicBlock(128, 128)
        self.pool = nn.AvgPool2d(8)
        self.linear = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.B3(self.B2(self.B1(self.B0(x))))
        out = self.pool(out)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        return final



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class ResNet(nn.Module):
    # feat_scale lets us deal with CelebA, other non-32x32 datasets
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1, dataset='cifar10'):
        super(ResNet, self).__init__()

        assert dataset in ['cifar10', 'celeba128']
        first_stride = {'cifar10': 1, 'celeba128': 2}[dataset]

        widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in widths]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=first_stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=first_stride)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool = nn.AvgPool2d(4)
        self.linear = nn.Linear(feat_scale * widths[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return SequentialWithArgs(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        assert (not no_relu), \
            "no_relu not yet supported for this architecture"
        out = F.relu(self.bn1(self.conv1(x)))
        # print('conv1 out shape: {}'.format(out.shape))
        out = self.layer1(out)
        # print('layer1 out shape: {}'.format(out.shape))
        out = self.layer2(out)
        # print('layer2 out shape: {}'.format(out.shape))
        out = self.layer3(out)
        # print('layer3 out shape: {}'.format(out.shape))
        out = self.layer4(out, fake_relu=fake_relu)
        # print('layer4 out shape: {}'.format(out.shape))
        out = self.pool(out)
        # print('avgpool out shape: {}'.format(out.shape))
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        if with_latent:
            return final, pre_out
        return final


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet18Wide(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], wd=1.5, **kwargs)


def ResNet18Thin(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], wd=.75, **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet50Wide(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], wm=1.5, **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


resnet50 = ResNet50
resnet50wide = ResNet50Wide
resnet18 = ResNet18
resnet101 = ResNet101
resnet152 = ResNet152


# resnet18thin = ResNet18Thin
# resnet18wide = ResNet18Wide
def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
