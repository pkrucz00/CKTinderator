
import math
from math import log2

from einops import rearrange, reduce, repeat
from kornia.filters import filter2d

import torch
from torch import nn, einsum
import torch.nn.functional as F

IMG_SIZE = 256
LATENT_DIM = 256

def exists(val):
    return val is not None

def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    return result * (1 if freq == 0 else math.sqrt(2))

def is_power_of_two(val):
    return log2(val).is_integer()

def default(val, d):
    return val if exists(val) else d

def get_dct_weights(width, channel, fidx_u, fidx_v):
    dct_weights = torch.zeros(1, channel, width, width)
    c_part = channel // len(fidx_u)

    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for x in range(width):
            for y in range(width):
                coor_value = get_1d_dct(x, u_x, width) * get_1d_dct(y, v_y, width)
                dct_weights[:, i * c_part: (i + 1) * c_part, x, y] = coor_value

    return dct_weights

def Conv2dSame(dim_in, dim_out, kernel_size, bias = True):
    pad_left = kernel_size // 2
    pad_right = (pad_left - 1) if (kernel_size % 2) == 0 else pad_left

    return nn.Sequential(
        nn.ZeroPad2d((pad_left, pad_right, pad_left, pad_right)),
        nn.Conv2d(dim_in, dim_out, kernel_size, bias = bias)
    )

class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, kernel_size = 3):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.kernel_size = kernel_size
        self.nonlin = nn.GELU()

        self.to_lin_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_lin_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding = 1, bias = False)

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias = False)

        self.to_out = nn.Conv2d(inner_dim * 2, dim, 1)

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]

        # linear attention

        lin_q, lin_k, lin_v = (self.to_lin_q(fmap), *self.to_lin_kv(fmap).chunk(2, dim = 1))
        lin_q, lin_k, lin_v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (lin_q, lin_k, lin_v))

        lin_q = lin_q.softmax(dim = -1)
        lin_k = lin_k.softmax(dim = -2)

        lin_q = lin_q * self.scale

        context = einsum('b n d, b n e -> b d e', lin_k, lin_v)
        lin_out = einsum('b n d, b d e -> b n e', lin_q, context)
        lin_out = rearrange(lin_out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        # conv-like full attention

        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) c x y', h = h), (q, k, v))

        k = F.unfold(k, kernel_size = self.kernel_size, padding = self.kernel_size // 2)
        v = F.unfold(v, kernel_size = self.kernel_size, padding = self.kernel_size // 2)

        k, v = map(lambda t: rearrange(t, 'b (d j) n -> b n j d', d = self.dim_head), (k, v))

        q = rearrange(q, 'b c ... -> b (...) c') * self.scale

        sim = einsum('b i d, b i j d -> b i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()

        attn = sim.softmax(dim = -1)

        full_out = einsum('b i j, b i j d -> b i d', attn, v)
        full_out = rearrange(full_out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        # add outputs of linear attention + conv like full attention

        lin_out = self.nonlin(lin_out)
        out = torch.cat((lin_out, full_out), dim = 1)
        return self.to_out(out)


class FCANet(nn.Module):
    def __init__(
        self,
        *,
        chan_in,
        chan_out,
        reduction = 4,
        width
    ):
        super().__init__()

        freq_w, freq_h = ([0] * 8), list(range(8)) # in paper, it seems 16 frequencies was ideal
        dct_weights = get_dct_weights(width, chan_in, [*freq_w, *freq_h], [*freq_h, *freq_w])
        self.register_buffer('dct_weights', dct_weights)

        chan_intermediate = max(3, chan_out // reduction)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = reduce(x * self.dct_weights, 'b c (h h1) (w w1) -> b c h1 w1', 'sum', h1 = 1, w1 = 1)
        return self.net(x)

class GlobalContext(nn.Module):
    def __init__(
        self,
        *,
        chan_in,
        chan_out
    ):
        super().__init__()
        self.to_k = nn.Conv2d(chan_in, 1, 1)
        chan_intermediate = max(3, chan_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        context = self.to_k(x)
        context = context.flatten(2).softmax(dim = -1)
        out = einsum('b i n, b c n -> b c i', context, x.flatten(2))
        out = out.unsqueeze(-1)
        return self.net(out)


class Noise(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise = None):
        b = x.shape[0]
        h = x.shape[2]
        w = x.shape[3] 
        device = x.device

        if not exists(noise):
            noise = torch.randn(b, 1, h, w, device = device)

        return x + self.weight * noise


class PixelShuffleUpsample(nn.Module):
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

    
class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2d(x, f, normalized=True)


class Generator(nn.Module):
    def __init__(
        self,
        *,
        image_size = IMG_SIZE,
        latent_dim = LATENT_DIM,
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        attn_res_layers = [32],
        freq_chan_attn = False,
    ):
        super().__init__()
        resolution = log2(image_size)
        assert is_power_of_two(image_size), 'image size must be a power of 2'

        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        fmap_max = default(fmap_max, latent_dim)

        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim * 2, 4),
            nn.BatchNorm2d(latent_dim * 2),
            nn.GLU(dim = 1)
        )

        num_layers = int(resolution) - 2
        features = list(map(lambda n: (n,  2 ** (fmap_inverse_coef - n)), range(2, num_layers + 2)))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))
        features = list(map(lambda n: 3 if n[0] >= 8 else n[1], features))
        features = [latent_dim, *features]

        in_out_features = list(zip(features[:-1], features[1:]))

        self.res_layers = range(2, num_layers + 2)
        self.layers = nn.ModuleList([])
        self.res_to_feature_map = dict(zip(self.res_layers, in_out_features))

        self.sle_map = ((3, 7), (4, 8), (5, 9), (6, 10))
        self.sle_map = list(filter(lambda t: t[0] <= resolution and t[1] <= resolution, self.sle_map))
        self.sle_map = dict(self.sle_map)

        self.num_layers_spatial_res = 1

        for (res, (chan_in, chan_out)) in zip(self.res_layers, in_out_features):
            image_width = 2 ** res
            attn = None
            if image_width in attn_res_layers:
                attn = PreNorm(chan_in, LinearAttention(chan_in))

            sle = None
            if res in self.sle_map:
                residual_layer = self.sle_map[res]
                sle_chan_out = self.res_to_feature_map[residual_layer - 1][-1]

                if freq_chan_attn:
                    sle = FCANet(
                        chan_in = chan_out,
                        chan_out = sle_chan_out,
                        width = 2 ** (res + 1)
                    )
                else:
                    sle = GlobalContext(
                        chan_in = chan_out,
                        chan_out = sle_chan_out
                    )

            layer = nn.ModuleList([
                nn.Sequential(
                    PixelShuffleUpsample(chan_in),
                    Blur(),
                    Conv2dSame(chan_in, chan_out * 2, 4),
                    Noise(),
                    nn.BatchNorm2d(chan_out * 2),
                    nn.GLU(dim = 1)
                ),
                sle,
                attn
            ])
            self.layers.append(layer)

        self.out_conv = nn.Conv2d(features[-1], init_channel, 3, padding = 1)

    def forward(self, x):
        x = rearrange(x, 'b c -> b c () ()')
        x = self.initial_conv(x)
        x = F.normalize(x, dim = 1)

        residuals = dict()

        for (res, (up, sle, attn)) in zip(self.res_layers, self.layers):
            if exists(attn):
                x = attn(x) + x

            x = up(x)

            if exists(sle):
                out_res = self.sle_map[res]
                residual = sle(x)
                residuals[out_res] = residual

            next_res = res + 1
            if next_res in residuals:
                x = x * residuals[next_res]

        return self.out_conv(x)