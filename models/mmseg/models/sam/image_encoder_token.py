
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock
import math
import warnings
from itertools import repeat
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs

from .basic import TD_Resblock, STD_Resblock
from .basic import TDifferenceConv, SDifferenceConv

class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:

            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        self.scale_factor = 24

        self.prompt_type = 'highpass'
        self.tuning_stage = 1234
        self.input_type = 'fft'
        self.freq_nums = 0.25
        self.handcrafted_tune = True
        self.embedding_tune = True
        self.adaptor = 'adaptor'
        self.prompt_generator = PromptGenerator(self.scale_factor, self.prompt_type, self.embed_dim,
                                                self.tuning_stage, self.depth,
                                                self.input_type, self.freq_nums,
                                                self.handcrafted_tune, self.embedding_tune, self.adaptor,
                                                img_size, patch_size)
        self.num_stages = self.depth
        self.out_indices = tuple(range(self.num_stages))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        multi_frame = x

        N, T, C, H, W = x.shape
        mid = T // 2
        x = x[:, mid, ...]
        x = self.patch_embed(x)

        embedding_feature = self.prompt_generator.init_embeddings(x)

        temporal_features, middle_mask = self.prompt_generator.init_temporal_features(multi_frame)

        temporal_token_embedding = self.prompt_generator.init_temporal_token_features(multi_frame)

        prompt = self.prompt_generator.get_prompt(temporal_features, embedding_feature)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        outs = []
        for i, blk in enumerate(self.blocks):

            x = prompt[i].reshape(B, H, W, -1) + x
            x = blk(x)
            if i in self.out_indices:
                outs.append(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x, temporal_token_embedding, middle_mask

def to_2tuple(x):
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, 2))

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """# type: (Tensor, float, float, float, float) -> Tensor"""
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):

        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():

        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor

class PromptGenerator(nn.Module):
    def __init__(self, scale_factor, prompt_type, embed_dim, tuning_stage, depth, input_type,
                 freq_nums, handcrafted_tune, embedding_tune, adaptor, img_size, patch_size):
        """
        Args:
        """
        super(PromptGenerator, self).__init__()
        self.scale_factor = scale_factor
        self.prompt_type = prompt_type
        self.embed_dim = embed_dim
        self.input_type = input_type
        self.freq_nums = freq_nums
        self.tuning_stage = tuning_stage
        self.depth = depth
        self.handcrafted_tune = handcrafted_tune
        self.embedding_tune = embedding_tune
        self.adaptor = adaptor

        self.shared_mlp = nn.Linear(self.embed_dim//self.scale_factor, self.embed_dim)
        self.embedding_generator = nn.Linear(self.embed_dim, self.embed_dim//self.scale_factor)
        for i in range(self.depth):
            lightweight_mlp = nn.Sequential(
                nn.Linear(self.embed_dim//self.scale_factor, self.embed_dim//self.scale_factor),
                nn.GELU()
            )
            setattr(self, 'lightweight_mlp_{}'.format(str(i)), lightweight_mlp)

        self.prompt_generator = PatchEmbed2(img_size=img_size,
                                                   patch_size=patch_size, in_chans=3,
                                                   embed_dim=self.embed_dim//self.scale_factor)

        self.mid_channel = 16

        self.temp_diff = HybridTemporalExtractor(in_chans=3,
                                               mid_chans=self.mid_channel,
                                               out_chans=self.mid_channel,
                                                global_mode="gru")

        self.temporal_patch_embed = PatchEmbed2(
            img_size=img_size//4, patch_size=4,
            in_chans=self.mid_channel, embed_dim=self.embed_dim // self.scale_factor
        )

        self.temporal_patch_embed_token = PatchEmbed2(
            img_size=img_size//4, patch_size=4,
            in_chans=self.mid_channel, embed_dim=256
        )

        self.middle_mask_decoder = nn.Sequential(
            nn.Conv2d(self.mid_channel, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_embeddings(self, x):
        N, C, H, W = x.permute(0, 3, 1, 2).shape
        x = x.reshape(N, C, H*W).permute(0, 2, 1)
        return self.embedding_generator(x)

    def init_handcrafted(self, x):
        x = self.fft(x, self.freq_nums)
        return self.prompt_generator(x)

    def init_temporal_features(self, x):
        '''
        args:
        input x: torch  N, T, 3, 1024, 1024
        output x: torch  N, 24, 64, 64
        '''

        x = self.temp_diff(x)
        middle_mask = self.middle_mask_decoder(x)
        x = self.temporal_patch_embed(x)
        return x, middle_mask

    def init_temporal_token_features(self, x):
        '''
        args:
        input x: torch  N, T, 3, 1024, 1024
        output x: torch  N, 24, 64, 64
        '''

        x = self.temp_diff(x)

        x = self.temporal_patch_embed_token(x)
        return x

    def get_prompt(self, handcrafted_feature, embedding_feature):
        N, C, H, W = handcrafted_feature.shape
        handcrafted_feature = handcrafted_feature.view(N, C, H*W).permute(0, 2, 1)
        prompts = []
        for i in range(self.depth):
            lightweight_mlp = getattr(self, 'lightweight_mlp_{}'.format(str(i)))

            prompt = lightweight_mlp(handcrafted_feature)
            prompts.append(self.shared_mlp(prompt))
        return prompts

    def forward(self, x):
        if self.input_type == 'laplacian':
            pyr_A = self.lap_pyramid.pyramid_decom(img=x, num=self.freq_nums)
            x = pyr_A[:-1]
            laplacian = x[0]
            for x_i in x[1:]:
                x_i = F.interpolate(x_i, size=(laplacian.size(2), laplacian.size(3)), mode='bilinear', align_corners=True)
                laplacian = torch.cat([laplacian, x_i], dim=1)
            x = laplacian
        elif self.input_type == 'fft':
            x = self.fft(x, self.freq_nums)
        elif self.input_type == 'all':
            x = self.prompt.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        prompt = self.prompt_generator(x)

        if self.mode == 'input':
            prompt = self.proj(prompt)
            return prompt
        elif self.mode == 'stack':
            prompts = []
            for i in range(self.depth):
                proj = getattr(self, 'proj_{}'.format(str(i)))
                prompts.append(proj(prompt))
            return prompts
        elif self.mode == 'hierarchical':
            prompts = []
            for i in range(self.depth):
                proj_prompt = getattr(self, 'proj_prompt_{}'.format(str(i)))
                prompt = proj_prompt(prompt)
                prompts.append(self.proj_token(prompt))
            return prompts

    def fft(self, x, rate):

        mask = torch.zeros(x.shape).to(x.device)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))

        fft = fft * (1 - mask)

        fr = fft.real
        fi = fft.imag

        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real

        inv = torch.abs(inv)

        return inv

try:
    TD = TDifferenceConv
    SD = SDifferenceConv
except Exception:
    TD = None
    SD = None

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv_z = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=pad)
        self.conv_r = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=pad)
        self.conv_h = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=pad)

    def forward(self, x, h):

        if h is None:
            h = torch.zeros(x.size(0), self.conv_z.out_channels, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        combined = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.conv_z(combined))
        r = torch.sigmoid(self.conv_r(combined))
        combined2 = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined2))
        h_new = (1 - z) * h + z * h_tilde
        return h_new

class HybridTemporalExtractor(nn.Module):
    def __init__(self, in_chans=3, mid_chans=8, out_chans=8, attn_embed=24, attn_heads=4, gru_hidden=32, patch_size=4, global_mode="gru"):
        super().__init__()
        self.in_chans = in_chans
        self.mid_chans = mid_chans
        self.out_chans = out_chans
        self.patch_size = patch_size
        self.global_mode = global_mode

        if TD is not None and SD is not None:
            self.front = nn.Sequential(
                TD(in_chans, mid_chans, kernel_size=(3,1,1), stride=(1,2,2), padding=(1,0,0)),
                nn.BatchNorm3d(mid_chans),
                nn.ReLU(inplace=True),
                SD(mid_chans, mid_chans, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
                nn.BatchNorm3d(mid_chans),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_chans, mid_chans*2, kernel_size=1, bias=False),
                nn.BatchNorm3d(mid_chans*2),
                nn.ReLU(inplace=True),
            )
            feat_c = mid_chans * 2
        else:
            self.front = nn.Sequential(
                nn.Conv3d(in_chans, mid_chans, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,2,2)),
                nn.BatchNorm3d(mid_chans),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_chans, mid_chans*2, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,2,2)),
                nn.BatchNorm3d(mid_chans*2),
                nn.ReLU(inplace=True),
            )
            feat_c = mid_chans * 2

        self.feat_c = feat_c

        self.attn_embed = attn_embed
        self.attn_heads = attn_heads
        self.patch_proj = nn.Conv2d(feat_c, attn_embed, kernel_size=1)
        self.attn_ln = nn.LayerNorm(attn_embed)
        self.mha = nn.MultiheadAttention(embed_dim=attn_embed, num_heads=attn_heads, batch_first=False)
        self.attn_out_proj = nn.Conv2d(attn_embed, out_chans, kernel_size=1)

        if self.global_mode == "gru":
            self.gru_input_proj = nn.Conv2d(feat_c, gru_hidden, kernel_size=3, padding=1)
            self.convgru_cell = ConvGRUCell(gru_hidden, gru_hidden, kernel_size=3)
            self.gru_out = nn.Conv2d(gru_hidden, out_chans, kernel_size=1)

        elif self.global_mode == "attn":

            assert gru_hidden % attn_heads == 0, "gru_hidden must be divisible by attn_heads"
            self.gru_input_proj = nn.Conv2d(feat_c, gru_hidden, kernel_size=1)
            self.global_ln = nn.LayerNorm(gru_hidden)
            self.global_mha = nn.MultiheadAttention(
                embed_dim=gru_hidden,
                num_heads=attn_heads,
                batch_first=False
            )
            self.gru_out = nn.Conv2d(gru_hidden, out_chans, kernel_size=1)

        else:
            raise ValueError(f"Unsupported global_mode: {self.global_mode}")

        self.motion_smooth = nn.Sequential(
            nn.Conv2d(feat_c, max(1, feat_c//2), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, feat_c//2), 1, kernel_size=1)
        )

        self.center_proj = nn.Conv2d(feat_c, out_chans, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_chans * 3 + 1, out_chans * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans * 2, out_chans, kernel_size=1)
        )

    def forward(self, x):

        N, T, C, H, W = x.shape
        xf = x.permute(0, 2, 1, 3, 4)
        xf = self.front(xf)
        _, feat_c, T2, H2, W2 = xf.shape
        if T2 != T:
            raise RuntimeError(f"front changed T dim: expected {T} got {T2}")
        frames = xf.permute(0, 2, 1, 3, 4).contiguous()

        p = self.patch_size
        patches = []
        for t in range(T):
            f = frames[:, t]
            f_proj = self.patch_proj(f)
            f_pooled = F.avg_pool2d(f_proj, kernel_size=p, stride=p)
            patches.append(f_pooled)
        patches = torch.stack(patches, dim=1)
        Np, T, E, Hp, Wp = patches.shape
        P = Hp * Wp

        patches_seq = patches.permute(1, 0, 3, 4, 2).contiguous().view(T, N * P, E)

        patches_seq = self.attn_ln(patches_seq)

        attn_out, _ = self.mha(patches_seq, patches_seq, patches_seq)
        attn_agg = attn_out.mean(dim=0).view(N, Hp, Wp, E).permute(0, 3, 1, 2).contiguous()
        attn_up = F.interpolate(attn_agg, scale_factor=p, mode='nearest')
        attn_branch = self.attn_out_proj(attn_up)

        if self.global_mode == "gru":
            h = None
            for t in range(T):
                f = frames[:, t]
                inp = self.gru_input_proj(f)
                h = self.convgru_cell(inp, h)
            gru_branch = self.gru_out(h)

        elif self.global_mode == "attn":

            seq = []
            for t in range(T):
                f = frames[:, t]
                inp = self.gru_input_proj(f)
                seq.append(inp)

            seq = torch.stack(seq, dim=1)
            N_, T_, G, H_, W_ = seq.shape

            seq = seq.permute(1, 0, 3, 4, 2).contiguous().view(T_, N_ * H_ * W_, G)
            seq = self.global_ln(seq)
            seq, _ = self.global_mha(seq, seq, seq)

            h = seq.mean(dim=0).view(N_, H_, W_, G).permute(0, 3, 1, 2).contiguous()
            gru_branch = self.gru_out(h)

        mid = T // 2
        center = frames[:, mid]
        others = [frames[:, t] for t in range(T) if t != mid]
        mean_others = torch.stack(others, dim=0).mean(dim=0) if len(others) > 0 else center

        num = (center * mean_others).sum(dim=1, keepdim=True)
        an = torch.norm(center, dim=1, keepdim=True)
        bn = torch.norm(mean_others, dim=1, keepdim=True)
        motion_map = num / (an * bn + 1e-6)
        motion_resp = 1.0 - motion_map.abs()
        motion_branch_small = self.motion_smooth(center)
        motion_final = torch.sigmoid(motion_resp + motion_branch_small)

        center_proj = self.center_proj(center)
        fusion_cat = torch.cat([attn_branch, gru_branch, center_proj, motion_final], dim=1)
        out = self.fuse(fusion_cat)
        return out

class PatchEmbed2(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)
        return x

class TemporalDiffExtractor(nn.Module):
    def __init__(self, in_chans=3, mid_chans=4, out_chans=8):
        super().__init__()

        self.conv_in = nn.Sequential(

            TDifferenceConv(in_chans, mid_chans, kernel_size=(3,1,1),stride=(1,2,2),
                            padding=(1,0,0), dilation=(1,1,1)),
            nn.BatchNorm3d(mid_chans),
            nn.ReLU(inplace=True),

            SDifferenceConv(mid_chans, mid_chans, kernel_size=(1,3,3),stride=(1,2,2),
                            padding=(0,1,1), dilation=(1,1,1)),
            nn.BatchNorm3d(mid_chans),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(
            STD_Resblock(mid_chans, mid_chans*2),
            STD_Resblock(mid_chans*2, out_chans)
        )

    def forward(self, x):

        x = x.permute(0, 2, 1, 3, 4)

        x = self.conv_in(x)
        x = self.layer1(x)

        mid_t = x.shape[2] // 2

        x = x[:, :, mid_t, :, :]

        return x

class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."

            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape

        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)

    if rel_pos.shape[0] != max_rel_dist:

        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)

        x = x.permute(0, 2, 3, 1)
        return x
