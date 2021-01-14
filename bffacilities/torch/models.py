import torch
import torch.nn as nn
import torch.nn.functional as F   # 神经网络模块中的常用功能 
from torch.jit.annotations import List, Tuple, Dict, Optional
from torch.nn import Parameter
from collections import OrderedDict

class ResidualModule(nn.Module):
    """
    Residual Module whose in_channels(default) is 256, out_channels(default) is 256

    Arguments:
        input(int): in_channels
        batch(bool): whether to deploy batch normlization layer
        stride(int): stride will be applied in the first ConvLayer
    """
    def __init__(self, in_channels=256, out_channels=256, div=2, \
            batch=False, **kwargs
        ):
        super().__init__()
        # self.name = name

        # padding = kwargs.get('padding', 0)
        mid_channels = out_channels // div

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) # conv3
        self.conv4 = nn.Conv2d(out_channels, mid_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(mid_channels,  mid_channels, kernel_size=3, padding=1) # padding with 0, to keep feature map size
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)  ## conv3
        self.batch = nn.BatchNorm2d(out_channels) if batch else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        out1 = self.relu(self.conv4(x))
        out1 = self.relu(self.conv5(out1))
        out1 = self.relu(self.conv3(out1))
        x = x + out1
        if self.batch is not None:
            x = self.batch(x)
        return x

    def __repr__(self):
        return f"Resnet-{self.name}"


class AttentionResidualModule(nn.Module):
    expansion = 4

    def __init__(self, in_channels=256, out_channels=256, **kwargs):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(out_channels, mid_channels, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.relu(self.conv0(x))
        out = self.relu(self.conv4(x))

        out = self.relu(self.conv5(out))

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        out += x
        out = self.relu(out)

        return out

class _HourGlassDownUnit(nn.Sequential):

    def __init__(self, in_channels, out_channels, block,\
        pool_kernel_size, pool_padding, **kwargs
        ):
        if pool_kernel_size is None:
            super().__init__(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            )
        else:
            super().__init__(
                # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                # ResidualModule(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                block(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(pool_kernel_size, stride=2, padding=pool_padding),
            )
    
class _HourGlassUpUnit(nn.Sequential):

    def __init__(self, in_channels, out_channels, block, mode='nearest', **kwargs):
        super().__init__(
            # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            # ResidualModule(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            block(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode=mode),
        )

class HourGlassModule(nn.Module):
    """Used for conv-deconv

    @Notes:
        Input feature map size should be greater than 32x32

    Arguments:
        in_channels: input feature map channels
        out_channels: output feature map channels

    Desc:
        channels will be always the same
    """

    def __init__(self, in_channels, out_channels, mid_channels=0, block=None, \
            debug=False, residual=False, **kwargs
        ):
        super().__init__()
        self.debug = debug
        if mid_channels == 0:
            mid_channels = out_channels
        # in_channels is 256, input feature map size is 32x32 / 64x64
        if block is None:
            block = ResidualModule if residual else nn.Conv2d
        kernel_size = (2, 2)
        padding = 0
        self.down_1 = _HourGlassDownUnit(in_channels, mid_channels, block=block, pool_kernel_size=kernel_size, pool_padding=padding, stride=2, batch=True)
        # 256 x 16 x 16  # 32 x 32
        self.down_2 = _HourGlassDownUnit(mid_channels, mid_channels, block=block, pool_kernel_size=kernel_size, pool_padding=padding, stride=2, batch=True)
        
        # 256 x 8 x 8
        self.down_3 = _HourGlassDownUnit(mid_channels, mid_channels, block=block, pool_kernel_size=kernel_size, pool_padding=padding, stride=2, batch=True)
        # 256 x 4 x 4   
        self.down_4 = _HourGlassDownUnit(mid_channels, mid_channels*2, block=block, pool_kernel_size=kernel_size, pool_padding=padding, stride=2, batch=True)
        # 256 x 2 x 2
        ###############################
        # 256 x 2 x2
        mode = kwargs.get('mode', 'nearest')
        self.up_4 = _HourGlassUpUnit(mid_channels*2, mid_channels, block=block, mode=mode)
        # 256 x 4 x 4
        self.up_3 = _HourGlassUpUnit(mid_channels, mid_channels, block=block, mode=mode)
        # 256 x 8 x 8
        self.up_2 = _HourGlassUpUnit(mid_channels, mid_channels, block=block, mode=mode)
        # 256 x 16 x 16
        self.up_1 = _HourGlassUpUnit(mid_channels, out_channels, block=block, mode=mode)
        # 256 x 32 x 32

    def forward(self, x):
        # print(x.shape)
        # 64 x 64
        dout1 = self.down_1(x)
        # 32 x 32
        dout2 = self.down_2(dout1)
        # 16 x 16
        dout3 = self.down_3(dout2)
        # 8 x 8
        dout4 = self.down_4(dout3)
        # 4 x 4

        # print(dout)
        out4 = F.relu(self.up_4(dout4))# + dout3)  # s3
        # 8
        out3 = F.relu(self.up_3(out4) + dout2) # s2
        # 16
        out2 = F.relu(self.up_2(out3) + dout1) # s1
        # 32
        out1 = F.relu(self.up_1(out2)) # + x) # according to paper HourGlass, x is not added
        # out1 = F.relu(self.up_1(out2) + x) # according to paper HourGlass, x is not added

        # 64
        if self.debug:
            return OrderedDict([
                ['0', out1], 
                ['1', out1], ['2', out2], ['3', out3],
                ['4', out4], ['5', dout4], ['6', dout3], ['7', dout2],
                # ['s3', s3], ['s2', s2], ['s1', s1]
            ])
        return out1

class HourGlassWrapper(nn.Module):
    """
    If last, the keypoint output will be returned instead of OrderedDict
    """
    def __init__(self, in_channels, out_channels, last=False, kp_output=True, **kwargs):
        super().__init__()
        self.last = last
        self.kp_output = True
        self.convDeconv = HourGlassModule(in_channels=in_channels, out_channels=in_channels, usePool=True, **kwargs)
        # self.res = ResidualModule(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.res = AttentionResidualModule(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.kpconv1 = nn.Conv2d(in_channels, out_channels, 1) # conv2
        if not self.last:
            self.conv = nn.Conv2d(in_channels, in_channels, 1) # conv3
            self.dconv2 = nn.Conv2d(out_channels, in_channels, 1) # conv3

    def forward(self, x, converge=True):
        uout1 = self.convDeconv(x)
        uout1 = self.res(uout1)

        if not self.kp_output:
            uout2 = F.relu(self.conv(uout1))
            return x + uout2

        out_kp1 = F.relu(self.kpconv1(uout1))
        # 4 x 32 x 32
        if not self.last:
            uout2 = F.relu(self.conv(uout1))
            if converge:
                dout2 = F.relu(self.dconv2(out_kp1))
                return OrderedDict([
                    ['0', x + uout2 + dout2], ['1', out_kp1]
                ])
            else:
                return OrderedDict([
                    ['0', x + uout2], ['1', out_kp1]
                ])
        return out_kp1

############## CapsNet ##################
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrimaryCaps(nn.Module):
    '''
    The `PrimaryCaps` layer consists of 32 capsule units. Each unit takes
    the output of the `Conv1` layer, which is a `[256, 20, 20]` feature
    tensor (omitting `batch_size`), and performs a 2D convolution with 8
    output channels, kernel size 9 and stride 2, thus outputing a [8, 6, 6]
    tensor. In other words, you can see these 32 capsules as 32 paralleled 2D
    convolutional layers. Then we concatenate these 32 capsules' outputs and
    flatten them into a tensor of size `[1152, 8]`, representing 1152 8D
    vectors, and send it to the next layer `DigitCaps`.

    As indicated in Section 4, Page 4 in the paper, *One can see PrimaryCaps
    as a Convolution layer with Eq.1 as its block non-linearity.*, outputs of
    the `PrimaryCaps` layer are squashed before being passed to the next layer.

    Reference: Section 4, Fig. 1
    '''

    def __init__(self, in_resolution=64):
        '''
        We build 8 capsule units in the `PrimaryCaps` layer, each can be
        seen as a 2D convolution layer.
        '''
        super(PrimaryCaps, self).__init__()
        self.resolution1 = in_resolution
        self.resolution2 = self.resolution1 // 2 - 4

        num_caps = 32
        self.resolution3 = (self.resolution2 ** 2) * num_caps

        self.capsules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=16, kernel_size=5, stride=2),
                nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0),
            )
            
            for i in range(num_caps)
        ])

    def forward(self, x):
        '''
        Each capsule outputs a [batch_size, 8, 6, 6] tensor, we need to
        flatten and concatenate them into a [batch_size, 8, 6*6, 32] size
        tensor and flatten and transpose into `u` [batch_size, 1152, 8], 
        where each [batch_size, 1152, 1] size tensor is the `u_i` in Eq.2. 

        #### Dimension transformation in this layer(ignoring `batch_size`):
        [256, 20, 20] --> [8, 6, 6] x 32 capsules --> [1152, 8]

        Note: `u_i` is one [1, 8] in the final [1152, 8] output, thus there are
        1152 `u_i`s.
        '''
        batch_size = x.size(0)

        u = []
        for i in range(32):
            # Input: [batch_size, 256, 20, 20]
            assert x.shape[-2:] == (self.resolution1, self.resolution1), f"{x.shape}, {self.resolution1}"

            u_i = self.capsules[i](x)
            assert u_i.shape[-3:] == (8, self.resolution2, self.resolution2)
            # u_i: [batch_size, 8, 6, 6]
            u_i = u_i.view(batch_size, 8, -1, 1) # 24 
            # u_i: [batch_size, 8, 36] # 576
            u.append(u_i)

        # u: [batch_size, 8, 36, 1] x 32
        u = torch.cat(u, dim=3)
        # u: [batch_size, 8, 36, 32]
        u = u.view(batch_size, 8, -1) # 576 * 32 == 
        # u: [batch_size, 8, 1152]
        u = torch.transpose(u, 1, 2)
        # u: [batch_size, 1152, 8]
        assert u.data.shape[-2:] == (self.resolution3, 8)

        # Squash before output
        u_squashed = self.squash(u)

        return u_squashed

    def squash(self, u):
        '''
        Args:
            `u`: [batch_size, 1152, 8]

        Return:
            `u_squashed`: [batch_size, 1152, 8]

        In CapsNet, we use the squash function after the output of both 
        capsule layers. Squash functions can be seen as activating functions
        like sigmoid, but for capsule layers rather than traditional fully
        connected layers, as they squash vectors instead of scalars.

        v_j = (norm(s_j) ^ 2 / (1 + norm(s_j) ^ 2)) * (s_j / norm(s_j))

        Reference: Eq.1 in Section 2.
        '''
        batch_size = u.size(0)

        # u: [batch_size, 1152, 8]
        square = u ** 2

        # square_sum for u: [batch_size, 1152]
        square_sum = torch.sum(square, dim=2)

        # norm for u: [batch_size, 1152]
        norm = torch.sqrt(square_sum)

        # factor for u: [batch_size, 1152]
        factor = norm ** 2 / (norm * (1 + norm ** 2))

        # u_squashed: [batch_size, 1152, 8]
        u_squashed = factor.unsqueeze(2) * u
        assert u_squashed.shape[-2:] == (self.resolution3, 8)

        return u_squashed


class Decoder(nn.Module):
    '''
    The decoder network consists of 3 fully connected layers. For each
    [10, 16] output, we mask out the incorrect predictions, and send
    the [16,] vector to the decoder network to reconstruct a [784,] size
    image.

    Reference: Section 4.1, Fig. 2
    '''

    def __init__(self, out_resolution, num_classes=10):
        '''
        The decoder network consists of 3 fully connected layers, with
        512, 1024, 784 neurons each.
        '''
        super().__init__()
        self.num_classes = num_classes

        self.fc1 = nn.Linear(16, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.out_resolution = out_resolution // 2
        self.final_resolution = out_resolution ** 2
        self.fc3 = nn.Linear(1024, self.out_resolution ** 2)
        self.up =  nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, v, target):
        '''
        Args:
            `v`: [batch_size, 10, 16]
            `target`: [batch_size, 10]

        Return:
            `reconstruction`: [batch_size, 784]

        We send the outputs of the `DigitCaps` layer, which is a
        [batch_size, 10, 16] size tensor into the decoder network, and
        reconstruct a [batch_size, 784] size tensor representing the image.
        '''
        batch_size = target.size(0)

        target = target.type(torch.FloatTensor)
        # mask: [batch_size, 10, 16]
        mask = torch.stack([target for i in range(16)], dim=2)
        assert mask.shape[-2:] == (10, 16)
        if v.device != torch.device("cpu"):
            mask = mask.to(v.device)

        # v: [bath_size, 10, 16]
        v_masked = mask * v
        v_masked = torch.sum(v_masked, dim=1)
        assert v_masked.shape[-1] == 16

        # Forward
        v = F.relu(self.fc1(v_masked))
        v = F.relu(self.fc2(v))
        reconstruction = torch.sigmoid(self.fc3(v))

        # assert reconstruction.shape[-1] == self.out_resolution
        reconstruction = reconstruction.view(batch_size, 1, self.out_resolution, self.out_resolution)
        reconstruction = self.up(reconstruction)
        return reconstruction # .view(batch_size, self.final_resolution)


class DigitCaps(nn.Module):
    '''
    The `DigitCaps` layer consists of 10 16D capsules. Compared to the traditional
    scalar output neurons in fully connected networks(FCN), the `DigitCaps` layer
    can be seen as an FCN with ten 16-dimensional output neurons, which we call
    these neurons "capsules".

    In this layer, we take the input `[1152, 8]` tensor `u` as 1152 [8,] vectors
    `u_i`, each `u_i` is a 8D output of the capsules from `PrimaryCaps` (see Eq.2
    in Section 2, Page 2) and sent to the 10 capsules. For each capsule, the tensor
    is first transformed by `W_ij`s into [1152, 16] size. Then we perform the Dynamic
    Routing algorithm to get the output `v_j` of size [16,]. As there are 10 capsules,
    the final output is [16, 10] size.

    #### Dimension transformation in this layer(ignoring `batch_size`):
    [1152, 8] --> [1152, 16] --> [1, 16] x 10 capsules --> [10, 16] output

    Note that in our codes we have vectorized these computations, so the dimensions
    above are just for understanding, actual dimensions of tensors are different.
    '''

    def __init__(self, routing, in_resolution, num_classes=10):
        '''
        There is only one parameter in this layer, `W` [1, 1152, 10, 16, 8], where
        every [8, 16] is a weight matrix W_ij in Eq.2, that is, there are 11520
        `W_ij`s in total.

        The the coupling coefficients `b` [64, 1152, 10, 1] is a temporary variable which
        does NOT belong to the layer's parameters. In other words, `b` is not updated
        by gradient back-propagations. Instead, we update `b` by Dynamic Routing
        in every forward propagation. See the docstring of `self.forward` for details.
        '''
        super(DigitCaps, self).__init__()
        self.routing = routing
        self.resolution1 = in_resolution
        self.num_classes = num_classes
        self.W = nn.Parameter(torch.randn(1, in_resolution, self.num_classes, 8, 16))

    def forward(self, u):
        '''
        Args:
            `u`: [batch_size, 1152, 8]
        Return:
            `v`: [batch_size, 10, 16]

        In this layer, we vectorize our computations by calling `W` and using
        `torch.matmul()`. Thus the full computaion steps are as follows.
            1. Expand `W` into batches and compute `u_hat` (Eq.2)
            2. Line 2: Initialize `b` into zeros
            3. Line 3: Start Routing for `r` iterations:
                1. Line 4: c = softmax(b)
                2. Line 5: s = sum(c * u_hat)
                3. Line 6: v = squash(s)
                4. Line 7: b += u_hat * v

        The coupling coefficients `b` can be seen as a kind of attention matrix
        in the attentional sequence-to-sequence networks, which is widely used in
        Neural Machine Translation systems. For tutorials on  attentional seq2seq
        models, see https://arxiv.org/abs/1703.01619 or
        http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

        Reference: Section 2, Procedure 1
        '''
        batch_size = u.size(0)

        # First, we need to expand the dimensions of `W` and `u` to compute `u_hat`
        assert u.shape[-2:] == (self.resolution1, 8)
        # u: [batch_size, 1152, 1, 1, 8]
        u = torch.unsqueeze(u, dim=2)
        u = torch.unsqueeze(u, dim=2)
        # Now we compute u_hat in Eq.2
        # u_hat: [batch_size, 1152, 10, 16]
        u_hat = torch.matmul(u, self.W).squeeze()

        # Line 2: Initialize b into zeros
        # b: [batch_size, 1152, 10, 1]
        b = torch.zeros(batch_size, self.resolution1, self.num_classes, 1)
        if b.device != u.device:
            b = b.to(u.device)

        # Start Routing
        for r in range(self.routing):
            # Line 4: c_i = softmax(b_i)
            # c: [b, 1152, 10, 1]
            c = F.softmax(b, dim=2)
            assert c.shape[-3:] == (self.resolution1, self.num_classes, 1)

            # Line 5: s_j = sum_i(c_ij * u_hat_j|i)
            # u_hat: [batch_size, 1152, 10, 16]
            # s: [batch_size, 10, 16]
            s = torch.sum(u_hat * c, dim=1)

            # Line 6: v_j = squash(s_j)
            # v: [batch_size, 10, 16]
            v = self.squash(s)
            assert v.shape[-2:] == ( self.num_classes, 16)

            # Line 7: b_ij += u_hat * v_j
            # u_hat: [batch_size, 1152, 10, 16]
            # v: [batch_size, 10, 16]
            # a: [batch_size, 10, 1152, 16]
            a = u_hat * v.unsqueeze(1)
            # b: [batch_size, 1152, 10, 1]
            b = b + torch.sum(a, dim=3, keepdim=True)

        return v

    def squash(self, s):
        '''
        Args:
            `s`: [batch_size, 10, 16]

        v_j = (norm(s_j) ^ 2 / (1 + norm(s_j) ^ 2)) * (s_j / norm(s_j))

        Reference: Eq.1 in Section 2.
        '''
        batch_size = s.size(0)

        # s: [batch_size, 10, 16]
        square = s ** 2

        # square_sum for v: [batch_size, 10]
        square_sum = torch.sum(square, dim=2)

        # norm for v: [batch_size, 10]
        norm = torch.sqrt(square_sum)

        # factor for v: [batch_size, 10]
        factor = norm ** 2 / (norm * (1 + norm ** 2))

        # v: [batch_size, 10, 16]
        v = factor.unsqueeze(2) * s
        assert v.shape[-2:] == (10, 16)

        return v


class CapsNet(nn.Module):

    def __init__(self, routing, num_classes = 10, **kwargs):
        '''
        The CapsNet consists of 3 layers: `Conv1`, `PrimaryCaps`, `DigitCaps`.`Conv1`
        is an ordinary 2D convolutional layer with 9x9 kernels, stride 2, 256 output
        channels, and ReLU activations. `PrimaryCaps` and `DigitCaps` are two capsule
        layers with Dynamic Routing between them. For further details of these two
        layers, see the docstrings of their classes. For each [1, 28, 28] input image,
        CapsNet outputs a [16, 10] tensor, representing the 16-dimensional output
        vector from 10 digit capsules.

        Reference: Section 4, Figure 1
        '''
        super().__init__()

        self.resolution1 = 28
        self.reconstruct_factor = 0.0005
        self.num_classes = num_classes
        # self.premodel =  nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=2), # 64
        #     nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=2, padding=2), # 32
        #     nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=0),
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
        # )
        
        self.premodel =  nn.Sequential(
            nn.Conv2d(in_channels=1,  out_channels=8,   kernel_size=5, stride=2, padding=0), # 62
            nn.Conv2d(in_channels=8,  out_channels=16,  kernel_size=3, stride=1, padding=0),  # 60
            nn.Conv2d(in_channels=16, out_channels=32,  kernel_size=5, stride=2, padding=0), # 28
            nn.Conv2d(in_channels=32, out_channels=256, kernel_size=3, stride=1, padding=1), # 28
        )
        # self.Conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9) # input_dim - 8
        self.primaryCaps = PrimaryCaps(in_resolution=self.resolution1)
        self.DigitCaps = DigitCaps(routing, self.primaryCaps.resolution3, num_classes=self.num_classes)

        self.Decoder = Decoder(self.resolution1, num_classes=self.num_classes)

    def forward(self, x):
        '''
        Args:
            `x`: [batch_size, 1, 28, 28] MNIST samples
        
        Return:
            `v`: [batch_size, 10, 16] CapsNet outputs, 16D prediction vectors of
                10 digit capsules

        The dimension transformation procedure of an input tensor in each layer:
            0. Input: [batch_size, 1, 28, 28] -->
            1. `Conv1` --> [batch_size, 256, 20, 20] --> 
            2. `PrimaryCaps` --> [batch_size, 8, 6, 6] x 32 capsules --> 
            3. Flatten, concatenate, squash --> [batch_size, 1152, 8] -->
            4. `W_ij`s and `DigitCaps` --> [batch_size, 16, 10] -->
            5. Length of 10 capsules --> [batch_size, 10] output probabilities
        '''
        # Input: [batch_size, 1, 28, 28]
        x = F.relu(self.premodel(x))
        # x = F.relu(self.Conv1(x))
        # PrimaryCaps input: [batch_size, 256, 20, 20]
        u = self.primaryCaps(x)
        # PrimaryCaps output u: [batch_size, 1152, 8] # 18432
        v = self.DigitCaps(u)
        # DigitCaps output v: [batsh_size, 10, 16]
        return v

    def marginal_loss(self, v, target, l=0.5):
        '''
        Args:
            `v`: [batch_size, 10, 16]
            `target`: [batch_size, 10]
            `l`: Scalar, lambda for down-weighing the loss for absent digit classes

        Return:
            `marginal_loss`: Scalar
        
        L_c = T_c * max(0, m_plus - norm(v_c)) ^ 2 + lambda * (1 - T_c) * max(0, norm(v_c) - m_minus) ^2
        
        Reference: Eq.4 in Section 3.
        '''
        batch_size = v.size(0)

        square = v ** 2
        square_sum = torch.sum(square, dim=2)
        # norm: [batch_size, 10]
        norm = torch.sqrt(square_sum)
        assert norm.size() == torch.Size([batch_size, 10])

        # The two T_c in Eq.4
        T_c = target.type(torch.FloatTensor)
        zeros = torch.zeros(norm.size())
        # Use GPU if available
        if v.device != torch.device("cpu"):
            zeros = zeros.to(v.device)
            T_c = T_c.to(v.device)

        # Eq.4
        marginal_loss = T_c * (torch.max(zeros, 0.9 - norm) ** 2) + \
            (1 - T_c) * l * (torch.max(zeros, norm - 0.1) ** 2)
        marginal_loss = torch.sum(marginal_loss)

        return marginal_loss

    def reconstruction_loss(self, reconstruction, image):
        '''
        Args:
            `reconstruction`: [batch_size, 784] Decoder outputs of images
            `image`: [batch_size, 1, 28, 28] MNIST samples

        Return:
            `reconstruction_loss`: Scalar Variable

        The reconstruction loss is measured by a squared differences
        between the reconstruction and the original image. 

        Reference: Section 4.1
        '''
        batch_size = image.size(0)
        # image: [batch_size, 784]
        image = F.interpolate(image, size=self.resolution1) # added by bf.
        # print(image.shape)
        image = image.unsqueeze(1)

        assert image.shape[-2:] == (self.resolution1, self.resolution1), image.shape
        
        # Scalar Variable
        reconstruction_loss = torch.sum((reconstruction - image) ** 2)
        return reconstruction_loss

    def loss(self, v, target, image):
        '''
        Args:
            `v`: [batch_size, 10, 16] CapsNet outputs
            `target`: [batch_size, 10] One-hot MNIST labels
            `image`: [batch_size, 1, 28, 28] MNIST samples

        Return:
            `L`: Scalar Variable, total loss
            `marginal_loss`: Scalar Variable
            `reconstruction_loss`: Scalar Variable

        The reconstruction loss is scaled down by 5e-4, serving as a
        regularization method.

        Reference: Section 4.1
        '''
        batch_size = image.size(0)

        marginal_loss = self.marginal_loss(v, target)

        # Get reconstructions from the decoder network
        reconstruction = self.Decoder(v, target)
        reconstruction_loss = self.reconstruction_loss(reconstruction, image)

        # Scalar Variable
        loss = (marginal_loss + self.reconstruct_factor * reconstruction_loss) / batch_size

        return loss, marginal_loss / batch_size, reconstruction_loss / batch_size
#######


# Resenet cbam

import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.conv2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.relu1(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu1(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
            x : B C H W
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        # self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2,
                               bias=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet34_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        try:
            model.load_state_dict(now_state_dict)
        except : pass
    return model


def resnet50_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        try:
            model.load_state_dict(now_state_dict, strict=False)
        except : pass
    return model


def resnet101_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet101'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet152_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model
