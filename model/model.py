import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from model.CAViT import SwinT

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class MRM(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type):
        super(MRM, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv1 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.kv2 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.mm1 = Modulation()
        self.mm2 = Modulation()
    def forward(self, x, y):
        x1 = self.norm1(x)
        y1 = self.norm1(y)
        b, c, h, w = x.shape
        kv1 = self.kv_dwconv1(self.kv1(x1))
        k1, v1 = kv1.chunk(2, dim=1)
        q1 = self.q_dwconv1(self.q1(y1))
        v1 = self.mm1(v1, x)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)
        out1 = (attn1 @ v1)
        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out1 = self.project_out1(out1)+y

        kv2 = self.kv_dwconv2(self.kv2(y1))
        k2, v2 = kv2.chunk(2, dim=1)
        q2 = self.q_dwconv2(self.q2(x1))
        v2 = self.mm2(v2, y)

        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.temperature
        attn2 = attn2.softmax(dim=-1)
        out2 = (attn2 @ v2)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out2 = self.project_out2(out2)+x
        return out1, out2

class SRM(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type):
        super(SRM, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qv1 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qv_dwconv1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_dwconv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.qv2 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qv_dwconv2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_dwconv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.mm1 = Modulation()
        self.mm2 = Modulation()
    def forward(self, x, y):
        x1 = self.norm1(x)
        y1 = self.norm1(y)
        b, c, h, w = x.shape

        qv1 = self.qv_dwconv1(self.qv1(x1))
        q1, v1 = qv1.chunk(2, dim=1)
        k1 = self.k_dwconv1(self.k1(y1))

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)
        out1 = (attn1 @ v1)
        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out1 = self.project_out1(out1)
        x_modulate = self.mm1(x, out1)
        out1 = out1 + x_modulate

        qv2 = self.qv_dwconv2(self.qv2(y1))
        q2, v2 = qv2.chunk(2, dim=1)
        k2 = self.k_dwconv2(self.k2(x1))

        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.temperature
        attn2 = attn2.softmax(dim=-1)
        out2 = (attn2 @ v2)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out2 = self.project_out2(out2)
        y_modulate = self.mm2(y, out2)
        out2 = out2 + y_modulate
        return out1, out2

class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out

class FIFM(nn.Module):
  def __init__(self, dim, kernel_size=3, padding=1, reduction=4):
      super(FIFM, self).__init__()
      self.convlayer1 = nn.Sequential(
          nn.Conv2d(dim // 2, dim // 2, kernel_size=3, stride=1, padding=1),
          nn.LeakyReLU(),
          nn.Conv2d(dim // 2, dim // 2, kernel_size=3, stride=1, padding=1),
          nn.LeakyReLU()
      )
      self.convlayer2 = nn.Sequential(
          nn.Conv2d(dim // 2, dim // 2, kernel_size=3, stride=1, padding=1),
          nn.LeakyReLU(),
          nn.Conv2d(dim // 2, dim // 2, kernel_size=3, stride=1, padding=1),
          nn.LeakyReLU()
      )

      self.x_ref_local_conv_1 = nn.Sequential(
          nn.Conv2d(dim, dim // reduction, 3, padding=1, bias=False),
          nn.BatchNorm2d(dim // reduction)
      )
      self.x_ref_local_conv_2 = nn.Sequential(
          nn.Conv2d(dim, dim//reduction, 5, padding=2, bias=False),
          nn.BatchNorm2d(dim//reduction)
      )
      self.x_ref_global_conv = nn.Sequential(
          nn.Conv2d(dim, dim // reduction, 7, padding=3, bias=False),
          nn.BatchNorm2d(dim // reduction)
      )
      self.x_ref_gap = _AsppPooling(dim, dim//reduction, nn.BatchNorm2d, norm_kwargs=None)
      self.x_ref_fuse = nn.Sequential(
          nn.Conv2d(4*dim//reduction, dim, kernel_size, padding=padding, bias=False),
          nn.BatchNorm2d(dim),
          nn.Sigmoid()
      )

      self.x_noref_local_conv_1 = nn.Sequential(
          nn.Conv2d(dim, dim // reduction, 3, padding=1, bias=False),
          nn.BatchNorm2d(dim // reduction)
      )
      self.x_noref_local_conv_2 = nn.Sequential(
          nn.Conv2d(dim, dim // reduction, 5, padding=2, bias=False),
          nn.BatchNorm2d(dim // reduction)
      )
      self.x_noref_global_conv = nn.Sequential(
          nn.Conv2d(dim, dim // reduction, 7, padding=3, bias=False),
          nn.BatchNorm2d(dim // reduction)
      )
      self.x_noref_gap = _AsppPooling(dim, dim // reduction, nn.BatchNorm2d, norm_kwargs=None)
      self.x_noref_fuse = nn.Sequential(
          nn.Conv2d(4*dim//reduction, dim, kernel_size, padding=padding, bias=False),
          nn.BatchNorm2d(dim),
          nn.Sigmoid()
      )
      self.softmax = nn.Softmax(dim=1)
      self.conv1x1 = nn.Conv2d(dim, dim // 2, kernel_size=1)

  def forward(self, x_ref, x_noref):
      f_ref = self.convlayer1(x_ref)
      f_noref = self.convlayer2(x_noref)
      f_ref = torch.cat([f_ref, x_noref], dim=1)
      f_noref = torch.cat([f_noref, x_ref], dim=1)

      feature_concat = torch.cat((x_ref, x_noref), dim=1)

      x_ref_weight = torch.cat((self.x_ref_local_conv_1(feature_concat), self.x_ref_local_conv_2(feature_concat),
                             self.x_ref_global_conv(feature_concat), self.x_ref_gap(feature_concat)), dim=1)
      x_ref_weight = self.x_ref_fuse(x_ref_weight).unsqueeze(1)

      x_noref_weight = torch.cat((self.x_noref_local_conv_1(feature_concat), self.x_noref_local_conv_2(feature_concat),
                              self.x_noref_global_conv(feature_concat), self.x_noref_gap(feature_concat)), dim=1)
      x_noref_weight = self.x_noref_fuse(x_noref_weight).unsqueeze(1)

      weights = self.softmax(torch.cat((x_ref_weight, x_noref_weight), dim=1))
      x_ref_weight, x_noref_weight = weights[:, 0:1, :, :, :].squeeze(1), weights[:, 1:2, :, :, :].squeeze(1)

      aggregated_feature = f_ref.mul(x_ref_weight)+f_noref.mul(x_noref_weight)
      aggregated_feature = self.conv1x1(aggregated_feature)

      return aggregated_feature

class ASM(nn.Module):
    def __init__(self, dim=128, num_heads=8, bias=False, LayerNorm_type='WithBias'):
        super(ASM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.conv1x1_1 = nn.Conv2d(dim//2, dim, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(dim//2, dim, kernel_size=1)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim//2, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.conv1x1 = nn.Conv2d(dim*2, dim, kernel_size=1)
        self.conv_cat = nn.Conv2d(dim, dim//2, kernel_size=1)
    def forward(self, x, ref):

        x1 = self.conv1x1_1(x)
        ref1 = self.conv1x1_2(ref)
        x1 = self.norm1(x1)
        ref1 = self.norm2(ref1)

        b, c, h, w = x1.shape
        kv = self.kv_dwconv(self.kv(x1))
        k, v = kv.chunk(2, dim=1)

        k = F.relu(k)
        v = -1*torch.min(v.float(), torch.tensor(0).float())
        k = self.conv1x1(torch.cat([k, v], dim=1))

        q = self.q_dwconv(self.q(ref1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out) + x

        out = self.conv_cat(torch.cat([ref, out], dim=1))
        return out

class Modulation(nn.Module):
    def __init__(self):
        super(Modulation, self).__init__()
        self.scale_conv0 = nn.Conv2d(64, 64, kernel_size=1)
        self.scale_conv1 = nn.Conv2d(64, 64, kernel_size=1)
        self.shift_conv0 = nn.Conv2d(64, 64, kernel_size=1)
        self.shift_conv1 = nn.Conv2d(64, 64, kernel_size=1)

    def forward(self, x, y):
        scale = self.scale_conv1(F.leaky_relu(self.scale_conv0(y), 0.1, inplace=True))
        shift = self.shift_conv1(F.leaky_relu(self.shift_conv0(y), 0.1, inplace=True))
        return x * (scale + 1) + shift

class Res_block(nn.Module):
    def __init__(self, nFeat, kernel_size=3, reduction=16):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(nFeat, nFeat//reduction, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nFeat // reduction, nFeat, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv2(self.relu(self.conv1(x)))
        pool = self.avg_pool(y)
        ca = self.ca(pool)
        out = y * ca
        out = out + x
        return out

class R_subblock(nn.Module):
    def __init__(self, nFeat, nDenselayer):
        super(R_subblock, self).__init__()
        modules = []
        for i in range(nDenselayer):
            modules.append(Res_block(nFeat))
        self.dense_layers = nn.Sequential(*modules)

    def forward(self, x):
        out = self.dense_layers(x)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channnel):
        super(Encoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channnel, 64, kernel_size=3, stride=1, padding=1),
            SwinT(n_feats=64),
        )
    def forward(self, x):
        out = self.layer(x)
        return out

class DEM(nn.Module):
    def __init__(self, embed_dim=64):
        super(DEM, self).__init__()
        self.srm = SRM(dim=embed_dim, num_heads=4, bias=False, LayerNorm_type='WithBias')
        self.mrm = MRM(dim=embed_dim, num_heads=4, bias=False, LayerNorm_type='WithBias')
        self.conv_layer = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.conv1x1 = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
    def forward(self, f, f2):
        fsrm1, fsrm2 = self.srm(f, f2)
        fmrm1, fmrm2 = self.mrm(f, f2)
        f_ref = self.conv_layer(torch.cat([fsrm1, fsrm2, fmrm1, fmrm2], dim=1))
        f_ref = self.conv1x1(torch.cat([f_ref, f2], dim=1))
        return f_ref


class Model(nn.Module):
    def __init__(self, embed_dim=64, nDenselayer=3):
        super(Model, self).__init__()
        ################################### Feature Extraction Network###################################
        self.conv_f1 = Encoder(in_channnel=6)
        self.conv_f2 = Encoder(in_channnel=6)
        self.conv_f3 = Encoder(in_channnel=6)
        self.conv_fs1 = Encoder(in_channnel=1)
        self.conv_fs2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        ################################### Feature Interaction Fusion ######################################
        self.dem1 = DEM()
        self.dem2 = DEM()

        self.fifm1 = FIFM(dim=embed_dim * 2)
        self.fifm2 = FIFM(dim=embed_dim * 2)
        self.conv_1x1 = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)

        self.asm1 = ASM()
        self.asm2 = ASM()

        ################################### HDR Reconstruction Network ####################################
        # Single Frame Reconstruction Network
        self.res_block1_1 = R_subblock(embed_dim, nDenselayer)
        self.res_block1_2 = R_subblock(embed_dim, nDenselayer)
        self.res_block1_3 = R_subblock(embed_dim, nDenselayer)
        self.resconv_1x1_1 = nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1)
        self.resconv_3x3_1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=True)
        self.conv_3x3_1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_last_1 = nn.Conv2d(embed_dim, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.act_last_1 = nn.Sigmoid()

        # Multi-exposure Fusion Reconstruction Network
        self.res_block2_1 = R_subblock(embed_dim, nDenselayer)
        self.res_block2_2 = R_subblock(embed_dim, nDenselayer)
        self.res_block2_3 = R_subblock(embed_dim, nDenselayer)
        self.resconv_1x1_2 = nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1)
        self.resconv_3x3_2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=True)
        self.conv_3x3_2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_last_2 = nn.Conv2d(embed_dim, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.act_last_2 = nn.Sigmoid()
    def forward(self, x1, x2, x3, xs):
        f1_1 = self.conv_f1(x1)
        f1_2 = self.conv_f2(x2)
        f1_3 = self.conv_f3(x3)
        fs_1 = self.conv_fs1(xs)

        f_ref1 = self.dem1(fs_1, f1_2)
        Ffifm1 = self.fifm1(f1_2, f1_1)
        Ffifm2 = self.fifm2(f1_2, f1_3)
        F1 = torch.cat([Ffifm1, Ffifm2], dim=1)
        Ffifm = self.conv_1x1(F1)
        Fasm1 = self.asm1(Ffifm, f_ref1)

        fs_2 = self.conv_fs2(fs_1)
        f_ref2 = self.dem2(fs_2, f_ref1)
        Fasm2 = self.asm2(Fasm1, f_ref2)

        out1_1 = self.res_block1_1(f_ref2)
        out1_2 = self.res_block1_2(out1_1)
        out1_3 = self.res_block1_3(out1_2)
        out1 = self.resconv_1x1_1(torch.cat([out1_1, out1_2, out1_3, f_ref2], dim=1))
        out1 = self.resconv_3x3_1(out1)
        res1 = self.conv_last_1(self.conv_3x3_1(out1 + f1_2))
        res1 = self.act_last_1(res1)

        out2_1 = self.res_block2_1(Fasm2)
        out2_2 = self.res_block2_2(out2_1)
        out2_3 = self.res_block2_3(out2_2)
        out2 = self.resconv_1x1_2(torch.cat([out2_1, out2_2, out2_3, Fasm2], dim=1))
        out2 = self.resconv_3x3_2(out2)
        res2 = self.conv_last_2(self.conv_3x3_2(out2 + f1_2))
        res2 = self.act_last_2(res2)
        return res1, res2