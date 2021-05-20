from collections import OrderedDict
import torch
import torch.nn as nn
from utils.util import b_split, b_merge
####################
# Basic blocks
####################


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ConcatBlock(nn.Module):
    # Concat the output of a submodule to its input
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr

def RCAN_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class ShortcutBlock_with2return(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock_with2return, self).__init__()
        self.sub = submodule

    def forward(self, x):
        fea1, fea2 = self.sub(x)
        output = x[0] + fea1
        return output, fea2

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr

class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)



def conv_block_IN(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, Instance normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wrong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)

def conv_block_downsample(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)

####################
# Useful blocks
####################


class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


class ResidualDenseBlock_5C(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nc+4*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x

class RRDB_catInput(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB_catInput, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc+1, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc+1, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc+1, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)

    def forward(self, x):
        # x_ddm = torch.cat([x[0], x[1]], dim=1)
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


class RRDB_Affine(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB_Affine, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.Aff1 = Affine_Module(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.Aff2 = Affine_Module(nc, kernel_size, gc, stride, bias, pad_type, \
                                  norm_type, act_type, mode)
        self.Aff3 = Affine_Module(nc, kernel_size, gc, stride, bias, pad_type, \
                                  norm_type, act_type, mode)

    def forward(self, x):
        x, ddm = x[0], x[1]
        out = self.RDB1(x)
        out = self.Aff1(out, ddm)
        out = self.RDB2(out)
        out = self.Aff2(out, ddm)
        out = self.RDB3(out)
        out = self.Aff3(out, ddm)
        return out.mul(0.2) + x, ddm

class RRDB_SEAN(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB_SEAN, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.SEAN = SEAN_resblk(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)

    def forward(self, x):
        x, ddm = x[0], x[1]
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        out, _ = self.SEAN(out, ddm)
        return out.mul(0.2) + x, ddm



class SEAN_resblk(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(SEAN_resblk, self).__init__()
        # self.SEAN1 = Affine_Module(nc, kernel_size, gc, stride, bias, pad_type, \
        #     norm_type, act_type, mode)
        branch1_m1, branch1_m2, branch2 = [], [], []
        branch1_m1.append(SEAN_Module(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode))
        branch1_m1.append(nn.LeakyReLU(inplace=True))
        branch1_m1.append(conv_block(nc, nc, kernel_size=3, stride=1, bias=True, pad_type='zero',
                                    norm_type=norm_type, act_type=act_type, mode='CNA'))


        branch1_m2.append(SEAN_Module(nc, kernel_size, gc, stride, bias, pad_type, \
                                      norm_type, act_type, mode))
        branch1_m2.append(nn.LeakyReLU(inplace=True))
        branch1_m2.append(conv_block(nc, nc, kernel_size=3, stride=1, bias=True, pad_type='zero',
                                     norm_type=norm_type, act_type=act_type, mode='CNA'))

        branch2.append(SEAN_Module(nc, kernel_size, gc, stride, bias, pad_type, \
                                   norm_type, act_type, mode))
        branch2.append(nn.LeakyReLU(inplace=True))
        branch2.append(conv_block(nc, nc, kernel_size=3, stride=1, bias=True, pad_type='zero',
                                  norm_type=norm_type, act_type=act_type, mode='CNA'))

        self.branch1_m1 = sequential(*branch1_m1)
        self.branch1_m2 = sequential(*branch1_m2)
        self.branch2 = sequential(*branch2)

    def forward(self, x, ddm):
        out = self.branch1_m1([x, ddm])
        out = self.branch1_m2([out, ddm])
        out = self.branch2([out, ddm])

        return out, ddm


class RRDB_ada(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB_ada, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.lda = nn.Parameter(torch.Tensor([0.4]), requires_grad=True)

    def forward(self, x):

        out = self.RDB1(x[0])
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(x[1] * self.lda) + x[0], x[1]

class RRDB_Residual_conv(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA', n_resconv=2, resconv_scale=[0.1, 1]):
        super(RRDB_Residual_conv, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.resconv_scale = resconv_scale
        # self.lda = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.res_conv = sequential(*[conv_block(nc, nc, kernel_size=3, stride=1, bias=True, pad_type='zero',
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(n_resconv)])


    def forward(self, x):

        out = self.RDB1(x[0])
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(x[1] * self.resconv_scale[1]) + self.res_conv(x[0]) * self.resconv_scale[0], x[1]

class RRDB_Residual_conv_concat(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA', n_resconv=2, n_ada_conv=2, adaptive_scale=[0.2, 1]):
        super(RRDB_Residual_conv_concat, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.adaptive_scale = adaptive_scale
        adaptive_conv_list = [conv_block(nc, nc, kernel_size=3, stride=1, bias=True, pad_type='zero',
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(n_ada_conv-1)]
        adaptive_conv_list = [conv_block(nc+1, nc, kernel_size=3, stride=1, bias=True, pad_type='zero',
            norm_type=norm_type, act_type=act_type, mode='CNA')] + adaptive_conv_list
        self.adaptive_conv = sequential(*adaptive_conv_list)
        # self.lda = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)

        resconv_list = [conv_block(nc, nc, kernel_size=3, stride=1, bias=True, pad_type='zero',
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(n_resconv-1)]
        resconv_list = [conv_block(nc+1, nc, kernel_size=3, stride=1, bias=True, pad_type='zero',
            norm_type=norm_type, act_type=act_type, mode='CNA')] + resconv_list

        self.res_conv = sequential(*resconv_list)
        # self.cprs_channel_conv = conv_block(nc+1, nc, kernel_size, stride, bias=bias, pad_type=pad_type, \
        #     norm_type=norm_type, act_type=act_type, mode=mode)

    def forward(self, x):
        input, ada_wegiths = x[0], x[1]
        out = self.RDB1(self.adaptive_conv(torch.cat((input, ada_wegiths * self.adaptive_scale[0]), dim=1)))
        out = self.RDB2(self.adaptive_conv(torch.cat((out, ada_wegiths * self.adaptive_scale[0]), dim=1)))
        out = self.RDB3(self.adaptive_conv(torch.cat((out, ada_wegiths * self.adaptive_scale[0]), dim=1)))
        residual = self.res_conv(torch.cat((input, ada_wegiths * self.adaptive_scale[1]), dim=1))
        return out.mul(0.2) + residual, ada_wegiths



class Affine_Module(nn.Module):
    def __init__(self, nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(Affine_Module, self).__init__()
        ddm_con_group1, ddm_con_group2 = [], []
        for i in range(2):
            if i == 0:
                ddm_con_group1.append(conv_block(1, nf, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                     norm_type=norm_type, act_type=act_type, mode=mode))
                ddm_con_group2.append(conv_block(1, nf, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                     norm_type=norm_type, act_type=act_type, mode=mode))
            else:
                ddm_con_group1.append(conv_block(nf, nf, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                     norm_type=norm_type, act_type=act_type, mode=mode))
                ddm_con_group2.append(conv_block(nf, nf, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                                 norm_type=norm_type, act_type=act_type, mode=mode))

        self.gamma1 = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)
        self.bias1 = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.ddm_conv1 = sequential(*ddm_con_group1)
        self.ddm_conv2 = sequential(*ddm_con_group2)

    def forward(self, x, ddm):
        df1 = self.ddm_conv1(ddm)
        df2 = self.ddm_conv1(ddm)
        x = self.gamma1 * df1 * x + self.bias1 * df2
        return x


class SEAN_Module(nn.Module):
    def __init__(self, nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(SEAN_Module, self).__init__()
        self.ddm_conv = conv_block(1, nf, kernel_size, stride, bias=bias, pad_type=pad_type, \
                             norm_type=norm_type, act_type=act_type, mode=mode)
        f_conv_rep_gamma = [conv_block(nf, nf, kernel_size, stride, bias=bias, pad_type=pad_type, \
                             norm_type=norm_type, act_type=act_type, mode=mode) for _ in range(2)]
        f_conv_rep_beta = [conv_block(nf, nf, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                      norm_type=norm_type, act_type=act_type, mode=mode) for _ in range(2)]
        f_conv_ddm_gamma = [conv_block(nf, nf, kernel_size, stride, bias=bias, pad_type=pad_type, \
                             norm_type=norm_type, act_type=act_type, mode=mode) for _ in range(2)]
        f_conv_ddm_beta = [conv_block(nf, nf, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                 norm_type=norm_type, act_type=act_type, mode=mode) for _ in range(2)]

        self.alpha_gamma = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.alpha_beta = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.f_conv_rep_gamma = sequential(*f_conv_rep_gamma)
        self.f_conv_rep_beta = sequential(*f_conv_rep_beta)
        self.f_conv_ddm_gamma = sequential(*f_conv_ddm_gamma)
        self.f_conv_ddm_beta = sequential(*f_conv_ddm_beta)

    def forward(self, x):
        x, ddm = x[0], x[1]
        ddm_repeat = ddm.repeat(1, 64, 1, 1)
        f_rep_gamma = self.f_conv_rep_gamma(ddm_repeat)
        f_rep_beta = self.f_conv_rep_gamma(ddm_repeat)

        ddm = self.ddm_conv(ddm)
        f_ddm_gamma = self.f_conv_ddm_gamma(ddm)
        f_ddm_beta = self.f_conv_ddm_beta(ddm)

        f_gamma_final = f_rep_gamma + (1 - self.alpha_gamma) * f_ddm_gamma
        f_beta_final = f_rep_beta + (1 - self.alpha_beta) * f_ddm_beta
        return x * f_gamma_final + f_beta_final


class Adaptive_Module(nn.Module):
    def __init__(self, nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA', ada_nb=4):
        super(Adaptive_Module, self).__init__()
        self.real_conv = [RRDB(nf, kernel_size) for _ in range(ada_nb)]
        self.fake_conv = [RRDB(nf, kernel_size) for _ in range(ada_nb)]
        self.real_conv = sequential(*self.real_conv)
        self.fake_conv = sequential(*self.fake_conv)

    def forward(self, x, mask):

        real_data, fake_data = b_split(x, mask)
        if len(real_data):
            real_data = self.real_conv(real_data)
        if len(fake_data):
            fake_data = self.fake_conv(fake_data)
        x = b_merge(real_data, fake_data, mask)
        return x


class RRDB_w_out(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB_w_out, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.conv1x1 = conv_block(64, 16, 1, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=act_type, mode=mode)

    def forward(self, x):
        out = self.RDB1(x[0])
        out = self.RDB2(out)
        out = self.RDB3(out)
        fea = self.conv1x1(out)
        return out.mul(0.2) + x[0], torch.cat((x[1], fea), dim=1)


class CARRDB(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA', conv=RCAN_conv, reduction=16):
        super(CARRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RCAB1 = Residual_Channle_Attention_Block(conv(nc, nc, kernel_size), nc, kernel_size, reduction)
        self.RCAB2 = Residual_Channle_Attention_Block(conv(nc, nc, kernel_size), nc, kernel_size, reduction)
        self.RCAB3 = Residual_Channle_Attention_Block(conv(nc, nc, kernel_size), nc, kernel_size, reduction)
        self.fusion_weightRRDB1 = torch.nn.Parameter(torch.tensor(0.1))
        self.fusion_weightRRDB2 = torch.nn.Parameter(torch.tensor(0.1))
        self.fusion_weightRRDB3 = torch.nn.Parameter(torch.tensor(0.1))

        self.fusion_weightRCAB1 = torch.nn.Parameter(torch.tensor(0.1))
        self.fusion_weightRCAB2 = torch.nn.Parameter(torch.tensor(0.1))
        self.fusion_weightRCAB3 = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        cab = self.RCAB1(x)
        out = self.RDB1(x)
        out = out * self.fusion_weightRRDB1 + cab * self.fusion_weightRCAB1

        cab = self.RCAB2(out)
        out = self.RDB2(out)
        out = out * self.fusion_weightRRDB2 + cab * self.fusion_weightRCAB2

        cab = self.RCAB3(out)
        out = self.RDB3(out)
        out = out * self.fusion_weightRRDB3 + cab * self.fusion_weightRCAB3
        return out + x

class CARRDBv2(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA', conv=RCAN_conv, reduction=16):
        super(CARRDBv2, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RCAB1 = Residual_Channle_Attention_Block(conv(nc, nc, kernel_size), nc, kernel_size, reduction)
        self.RCAB2 = Residual_Channle_Attention_Block(conv(nc, nc, kernel_size), nc, kernel_size, reduction)
        self.RCAB3 = Residual_Channle_Attention_Block(conv(nc, nc, kernel_size), nc, kernel_size, reduction)


    def forward(self, x):
        out = self.RDB1(x)
        out = self.RCAB1(out)
        out = self.RDB2(out)
        out = self.RCAB2(out)
        out = self.RDB3(out)
        out = self.RCAB3(out)
        return out.mul(0.2) + x


class CARRDBv3(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA', conv=RCAN_conv, reduction=16):
        super(CARRDBv3, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RCAB1 = Residual_Channle_Attention_Block(conv(nc, nc, kernel_size), nc, kernel_size, reduction)
        self.RCAB2 = Residual_Channle_Attention_Block(conv(nc, nc, kernel_size), nc, kernel_size, reduction)
        self.RCAB3 = Residual_Channle_Attention_Block(conv(nc, nc, kernel_size), nc, kernel_size, reduction)


    def forward(self, x):
        out = self.RDB1(x)
        out = self.RCAB1(out)
        out = self.RDB2(out)
        out = self.RCAB2(out)
        out = self.RDB3(out)
        out = self.RCAB3(out)
        return out.mul(0.2) + x


# # Channel Attention (CA) Layer
# class CALayerv3(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(CALayer, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#                 nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#                 nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y
#
# ## Residual Channel Attention Block (RCAB)
# class Residual_Channle_Attention_Blockv3(nn.Module):
#     def __init__(
#         self, conv, n_feat, kernel_size, reduction,
#         bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
#
#         super(Residual_Channle_Attention_Block, self).__init__()
#
#         modules_body = []
#         for i in range(2):
#             # modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#             modules_body.append(conv)
#             if bn: modules_body.append(nn.BatchNorm2d(n_feat))
#             if i == 0: modules_body.append(act)
#         modules_body.append(CALayer(n_feat, reduction))
#         self.body = nn.Sequential(*modules_body)
#         self.res_scale = res_scale
#
#     def forward(self, x):
#         res = self.body(x)
#         #res = self.body(x).mul(self.res_scale)
#         res += x
#         return res

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class Residual_Channle_Attention_Block(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(Residual_Channle_Attention_Block, self).__init__()

        modules_body = []
        for i in range(2):
            # modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            modules_body.append(conv)
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res




####################
# Upsampler
####################


def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                        pad_type='zero', norm_type=None, act_type='relu'):
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)

