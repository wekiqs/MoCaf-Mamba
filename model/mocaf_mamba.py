"""
IMFuse-Mamba Lite-V2C — 强约束特征补全版本
保持核心设计不变：多分支编码、Mamba SSM、形变对齐、mask-aware prompt restoration、dual decoder。

改动:
  encoder dims:  [12,24,48,96,384] → [12,20,32,64,192]
                 保留浅层细节表达，压缩深层冗余。
  decoder_base:  12 (解码器保持原宽度)
  d_state:       8 → 6 (恢复部分 Mamba SSM 容量)
  expand:        2 → 1 (保持 Mamba 内部轻量)
  scan_modes[4]: [0,0,0] (恢复最深 stage 三次扫描)
  rank_ratio:    4 → 8 (Token 分解更轻量)
  num_heads:     4 (恢复 deformable fusion 多头对齐能力)
  + enc→dec 1×1 投影层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch.cuda.amp import autocast
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from torch.utils.checkpoint import checkpoint

from .layers import general_conv3d_prenorm, fusion_prenorm
from .utils.initialization import InitWeights_He
from itertools import combinations

# ========================================================================
# 轻量化超参数
# ========================================================================
basic_dims = 12               # stem 保持原始浅层宽度
encoder_dims = [12, 24, 48, 96, 192]
decoder_base = 12             # 解码器通道基数 (保持原 12，保 dice)
transformer_basic_dims = 128
mlp_dim = 4096
num_heads = 8
depth = 1
patch_size = 8
input_patch_size = 128


class DropPath(nn.Module):
    """Stochastic Depth per sample (dropping entire residual branches)."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


class Softmax_32(nn.Module):
    def __init__(self):
        super(Softmax_32, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        x = self.softmax(x)
        return x


def compute_flow_with_checkpoint(flow_net, x):
    """使用梯度检查点计算流场"""

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(inputs[0])

        return custom_forward

    return checkpoint(create_custom_forward(flow_net), x, use_reentrant=False)


class BidirectionalMambaBlocks(nn.Module):
    def __init__(self, d_model, d_state=6, d_conv=4, expand=1, drop_path=0.1):
        """Lite-V2: d_state 8→6, expand 2→1"""
        super().__init__()
        self.d_model = d_model

        d_inner = d_model // 2
        self.mamba_fwd = Mamba(d_model=d_inner, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_inner, d_state=d_state, d_conv=d_conv, expand=expand)

        self.norm = nn.LayerNorm(d_model)
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.drop_path = DropPath(drop_path)

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        residual = x
        x = self.norm(x)
        x = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x_fwd, x_bwd = torch.split(x, self.d_model // 2, dim=-1)
        out_fwd = self.mamba_fwd(x_fwd)
        out_bwd = self.mamba_bwd(x_bwd.flip(dims=[1])).flip(dims=[1])
        out = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.dropout(out)
        out = self.drop_path(out)

        return out + residual


class MambaLayer_image(nn.Module):
    def __init__(self, channels, scan_modes=None, d_state=6, drop_path_rates=None):
        """Lite-V2: d_state 8→6"""
        super().__init__()
        self.channels = channels

        if scan_modes is None:
            scan_modes = [0, 1, 2]

        self.scan_modes = []
        for mode in scan_modes:
            if isinstance(mode, str):
                mode_map = {'DHW': 0, 'HWD': 1, 'WDH': 2}
                self.scan_modes.append(mode_map[mode.upper()])
            else:
                self.scan_modes.append(mode)

        self.depth = len(self.scan_modes)

        if drop_path_rates is None:
            drop_path_rates = [0.1] * self.depth
        elif len(drop_path_rates) != self.depth:
            raise ValueError(f"drop_path_rates length {len(drop_path_rates)} != depth {self.depth}")

        self.layers = nn.ModuleList([
            BidirectionalMambaBlocks(d_model=channels, d_state=d_state, drop_path=drop_path_rates[i])
            for i in range(self.depth)
        ])

        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, channels) * 1e-5)
            for _ in range(self.depth)
        ])

    def forward(self, x):
        assert x.ndim == 5, f"Expected 5D input [B, C, D, H, W], got {x.shape}"
        B, C, D, H, W = x.shape
        assert C == self.channels

        current_feat = x
        for i, layer in enumerate(self.layers):
            scan_mode = self.scan_modes[i]

            if scan_mode == 0:  # DHW
                x_permuted = current_feat
                feat_size = (D, H, W)
                inverse_permute = None
            elif scan_mode == 1:  # HWD
                x_permuted = current_feat.permute(0, 1, 3, 4, 2)
                feat_size = (H, W, D)
                inverse_permute = (0, 1, 4, 2, 3)
            else:  # scan_mode == 2, WDH
                x_permuted = current_feat.permute(0, 1, 4, 2, 3)
                feat_size = (W, D, H)
                inverse_permute = (0, 1, 3, 4, 2)

            x_flat = x_permuted.reshape(B, C, -1).transpose(-1, -2)
            x_mamba = layer(x_flat)
            x_out = x_mamba.transpose(-1, -2).reshape(B, C, *feat_size)

            if inverse_permute is not None:
                x_out = x_out.permute(*inverse_permute)

            current_feat = x_out * self.layer_scales[i].view(1, -1, 1, 1, 1)

        return current_feat + x


class SpatialChannelFactorizedToken(nn.Module):
    """空间-通道因子分解 (LoRA风格) 生成全尺寸 Token。"""

    def __init__(self, dim, spatial_size, rank_ratio=8):
        """Light: rank_ratio 4→8"""
        super().__init__()
        self.rank = max(1, dim // rank_ratio)
        self.spatial_size = spatial_size

        self.spatial_basis = nn.Parameter(torch.randn(1, self.rank, *spatial_size))
        self.mixer = nn.Conv3d(self.rank, dim, kernel_size=1, bias=False)

        nn.init.normal_(self.spatial_basis, std=0.02)
        nn.init.normal_(self.mixer.weight, std=0.02)

    def forward(self, B):
        token = self.mixer(self.spatial_basis)
        token = token.unsqueeze(1)
        token = token.expand(B, -1, -1, *self.spatial_size)
        return token


class DeformableAlignmentBlock(nn.Module):
    def __init__(self, in_channels, num_modals, spatial_size, max_displacement=1.0 / 32.0):
        super().__init__()
        self.max_displacement = max_displacement
        self.use = True

        hidden = num_modals * 4
        gn_groups = 4 if hidden % 4 == 0 else (2 if hidden % 2 == 0 else 1)
        self.offset_conv = nn.Sequential(
            nn.Conv3d(in_channels * num_modals, hidden, kernel_size=5, bias=True, padding=2),
            nn.GroupNorm(num_groups=gn_groups, num_channels=hidden),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, bias=True, padding=1),
            nn.GroupNorm(num_groups=gn_groups, num_channels=hidden),
            nn.GELU(),
            nn.Conv3d(hidden, num_modals * 3, kernel_size=3, bias=True, padding=1),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        nn.init.constant_(self.offset_conv[-1].weight, 0)
        nn.init.constant_(self.offset_conv[-1].bias, 0)

    def forward(self, x):
        B, M, C, H, W, D = x.shape
        x_concat = x.view(B, M * C, H, W, D)

        offsets = torch.tanh(self.offset_conv(x_concat)) * self.max_displacement
        offsets = offsets.view(B * M, 3, H, W, D)

        base_grid = self._generate_grid(D, H, W, x.device).unsqueeze(0).expand(B * M, -1, -1, -1, -1)
        sampling_grid = base_grid + offsets.permute(0, 2, 3, 4, 1)

        x_reshaped = x.view(B * M, C, H, W, D)
        x_aligned = F.grid_sample(x_reshaped, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)

        return x_aligned.view(B, M, C, H, W, D)

    def _generate_grid(self, D, H, W, device):
        d = torch.linspace(-1, 1, D, device=device)
        h = torch.linspace(-1, 1, H, device=device)
        w = torch.linspace(-1, 1, W, device=device)
        mesh = torch.meshgrid(d, h, w, indexing='ij')
        return torch.stack([mesh[2], mesh[1], mesh[0]], dim=-1)


class ParallelDeformableFusion(nn.Module):
    def __init__(self, dim, spatial_size, num_modals=4, rank_ratio=8, num_heads=4):
        """Lite-V2: rank_ratio 4→8, num_heads=4"""
        super().__init__()
        self.dim = dim
        self.num_modals = num_modals
        self.total_units = num_modals + 1

        if dim % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        s_size = spatial_size[0] if isinstance(spatial_size, (tuple, list)) else spatial_size

        if s_size >= 64:
            k, d = 5, 1
        elif s_size >= 32:
            k, d = 5, 1
        elif s_size >= 16:
            k, d = 3, 1
        else:
            k, d = 3, 1

        p = (k - 1) * d // 2

        self.deformable_align = DeformableAlignmentBlock(dim, num_modals, spatial_size, max_displacement=1.0 / 32)
        self.token_generator = SpatialChannelFactorizedToken(dim, spatial_size, rank_ratio)

        self.norm1 = nn.GroupNorm(1, dim)
        self.norm2 = nn.GroupNorm(1, dim)

        self.to_q = nn.Conv3d(
            self.total_units * dim, self.total_units * dim,
            kernel_size=3, padding=1, dilation=d,
            groups=self.total_units, bias=False
        )
        self.to_k = nn.Conv3d(
            self.total_units * dim, self.total_units * dim,
            kernel_size=3, padding=1, dilation=d,
            groups=self.total_units, bias=False
        )
        self.to_v = nn.Conv3d(
            self.total_units * dim, self.total_units * dim,
            kernel_size=3, padding=1, dilation=d,
            groups=self.total_units, bias=False
        )

        self.proj = nn.Conv3d(dim, dim, kernel_size=1)

        self.ffn = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=1)
        )

        self.cnn_branch = nn.Sequential(
            nn.Conv3d(dim * num_modals, dim * 2, kernel_size=k, padding=p, dilation=d),
            nn.GroupNorm(8, dim * 2),
            nn.GELU(),
            nn.Conv3d(dim * 2, dim, kernel_size=k, padding=p, dilation=d)
        )
        self.alpha = nn.Parameter(torch.ones(1))

        self.token_bias = nn.Parameter(torch.zeros(2))

    def forward(self, x, mask=None):
        B, MC, H, W, D = x.shape
        M = self.num_modals
        C = self.dim

        x_reshaped = x.view(B, M, C, H, W, D)
        x_aligned = self.deformable_align(x_reshaped)

        fused_token = self.token_generator(B)

        x_all = torch.cat([x_aligned, fused_token], dim=1)
        x_grouped_in = x_all.view(B, (M + 1) * C, H, W, D)

        q_grouped = self.to_q(x_grouped_in)
        k_grouped = self.to_k(x_grouped_in)
        v_grouped = self.to_v(x_grouped_in)

        Hh = self.num_heads
        Hd = self.head_dim
        q = q_grouped.view(B, M + 1, Hh, Hd, H, W, D)
        k = k_grouped.view(B, M + 1, Hh, Hd, H, W, D)
        v = v_grouped.view(B, M + 1, Hh, Hd, H, W, D)

        scale = Hd ** -0.5
        attn = torch.einsum('bukcxyz, bvkcxyz -> bukvxyz', q, k) * scale

        if mask is not None:
            present_mask = mask.type_as(attn)
            modality_bias = torch.where(
                present_mask > 0.5,
                self.token_bias[0].expand_as(present_mask),
                self.token_bias[1].expand_as(present_mask),
            )
            fused_bias = self.token_bias[0].view(1, 1).expand(B, 1)
            bias_full = torch.cat([modality_bias, fused_bias], dim=1)
            attn = attn + bias_full.view(B, 1, 1, M + 1, 1, 1, 1)

        attn = attn.softmax(dim=3)
        out = torch.einsum('bukvxyz, bvkcxyz -> bukcxyz', attn, v)
        out = out.reshape(B, M + 1, self.dim, H, W, D)

        out = out.reshape(B * (M + 1), C, H, W, D)
        x_shared_in = x_all.view(B * (M + 1), C, H, W, D)

        out = self.proj(out)
        out = self.norm1(x_shared_in + out)

        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)

        out = out.view(B, M + 1, C, H, W, D)

        fused_out = out[:, -1, ...]

        conv_input = x_aligned.view(B, M * C, H, W, D)
        out_conv_fusion = self.cnn_branch(conv_input)

        out_fused = fused_out + self.alpha * out_conv_fusion

        return out_fused


class MlpChannel(nn.Module):
    """ConvFFN: Conv1x1 -> 3x3 DW Conv -> GELU -> Conv1x1"""
    def __init__(self, hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.dw_conv = nn.Conv3d(mlp_dim, mlp_dim, kernel_size=3, padding=1, groups=mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dw_conv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU(inplace=True)

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU(inplace=True)

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU(inplace=True)

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual


class StemDWResidual(nn.Module):
    def __init__(self, in_channels=1, out_channels=basic_dims, norm="gn"):
        super().__init__()

        def Norm(c):
            if norm == "in":
                return nn.InstanceNorm3d(c, affine=True)
            return nn.GroupNorm(num_groups=4 if c % 4 == 0 else 1, num_channels=c)

        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

        self.block = nn.Sequential(
            Norm(out_channels),
            nn.GELU(),
            nn.Conv3d(
                out_channels, out_channels,
                kernel_size=5, padding=2,
                groups=out_channels,
            ),
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
        )

        self.out_norm = Norm(out_channels)
        self.out_act = nn.GELU()

    def forward(self, x):
        y = self.proj(x)
        y = y + self.block(y)
        out = self.out_act(self.out_norm(y))

        return out


class MambaEncoder(nn.Module):
    def __init__(self, scan_modes=[[], [0], [0], [0], [0]], dims=None, res_block=True, spatial_dims=3):
        """Lite-V2: asymmetric encoder dims [12,20,32,64,192], 1-direction scan only."""
        super().__init__()
        if dims is None:
            dims = encoder_dims
        self.stem = StemDWResidual()
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()

        total_blocks = sum(len(scan_modes[i]) for i in range(1, 5))
        max_dpr = 0.15
        if total_blocks > 1:
            dpr_list = [x.item() for x in torch.linspace(0.0, max_dpr, total_blocks)]
        else:
            dpr_list = [max_dpr] * total_blocks
        dpr_idx = 0
        stage_dprs = []
        for i in range(1, 5):
            n_blocks = len(scan_modes[i])
            stage_dprs.append(dpr_list[dpr_idx:dpr_idx + n_blocks])
            dpr_idx += n_blocks

        for i in range(1, 5):
            self.downsample_layers.append(nn.Sequential(
                nn.InstanceNorm3d(dims[i - 1]),
                nn.Conv3d(dims[i - 1], dims[i], kernel_size=2, stride=2),
            ))
            self.stages.append(MambaLayer_image(channels=dims[i], scan_modes=scan_modes[i], d_state=6,
                                                 drop_path_rates=stage_dprs[i - 1]))
            self.gscs.append(GSC(dims[i]))

        self.mlps = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.out_res_scales = nn.ParameterList()
        for i_layer in range(5):
            self.norm.append(nn.InstanceNorm3d(dims[i_layer]))
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))
            self.out_res_scales.append(nn.Parameter(torch.zeros(1, dims[i_layer], 1, 1, 1)))

    def forward(self, x):
        outs = []

        with autocast(enabled=True):
            x = self.stem(x)
            x_out = self.norm[0](x)
            x_out = self.mlps[0](x_out)
            x_out = x_out + self.out_res_scales[0] * x
        x, x_out = x.float(), x_out.float()
        outs.append(x_out)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)
            x_out = self.norm[i + 1](x)
            x_out = self.mlps[i + 1](x_out)
            x_out = x_out + self.out_res_scales[i + 1] * x
            outs.append(x_out)

        return tuple(outs)


# ========================================================================
# Decoders — 使用 decoder_base 保持原始宽度
# ========================================================================

class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4, mamba_skip=False, num_modals=4):
        super(Decoder_sep, self).__init__()

        norm_name = "instance"
        res_block: bool = True
        spatial_dims = 3

        self.decoderh = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=decoder_base * 32,
            out_channels=decoder_base * 16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=decoder_base * 16,
            out_channels=decoder_base * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=decoder_base * 8,
            out_channels=decoder_base * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=decoder_base * 4,
            out_channels=decoder_base * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = nn.ConvTranspose3d(
            in_channels=decoder_base * 2,
            out_channels=decoder_base * 2,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=True,
        )
        self.decoder_end = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=decoder_base * 1,
            out_channels=decoder_base * 2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out_end = UnetOutBlock(spatial_dims=spatial_dims, in_channels=decoder_base * 2, out_channels=num_cls)
        self.softmax = Softmax_32()
        self.mamba_skip = mamba_skip

    def forward(self, x1, x2, x3, x4, x5):
        dech = self.decoderh(x5)
        dec3 = self.decoder4(dech, x4)
        dec2 = self.decoder3(dec3, x3)
        dec1 = self.decoder2(dec2, x2)
        dec0 = self.decoder1(dec1)
        dec_end = dec0
        pred = self.softmax(self.out_end(dec_end))

        return pred


class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=4, mamba_skip=False, num_modals=4):
        super(Decoder_fuse, self).__init__()

        norm_name = "instance"
        res_block: bool = True
        spatial_dims = 3

        self.decoderh = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=decoder_base * 32,
            out_channels=decoder_base * 16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=decoder_base * 16,
            out_channels=decoder_base * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=decoder_base * 8,
            out_channels=decoder_base * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=decoder_base * 4,
            out_channels=decoder_base * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = nn.ConvTranspose3d(
            in_channels=decoder_base * 2,
            out_channels=decoder_base * 2,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=True,
        )
        self.decoder_end = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=decoder_base * 2,
            out_channels=decoder_base * 2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out_8 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=decoder_base * 16, out_channels=num_cls)
        self.out_4 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=decoder_base * 8, out_channels=num_cls)
        self.out_2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=decoder_base * 4, out_channels=num_cls)
        self.out_1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=decoder_base * 2, out_channels=num_cls)
        self.out_end = UnetOutBlock(spatial_dims=spatial_dims, in_channels=decoder_base * 2, out_channels=num_cls)

        self.softmax = Softmax_32()
        self.mamba_skip = mamba_skip

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        self.skip_scale2 = nn.Parameter(torch.zeros(1, decoder_base * 2, 1, 1, 1))
        self.skip_scale3 = nn.Parameter(torch.zeros(1, decoder_base * 4, 1, 1, 1))
        self.skip_scale4 = nn.Parameter(torch.zeros(1, decoder_base * 8, 1, 1, 1))

    def forward(self, x1, x2, x3, x4, x5):
        x2 = x2 * (1 + self.skip_scale2)
        x3 = x3 * (1 + self.skip_scale3)
        x4 = x4 * (1 + self.skip_scale4)

        dech = self.decoderh(x5)
        dec3 = self.decoder4(dech, x4)
        dec2 = self.decoder3(dec3, x3)
        dec1 = self.decoder2(dec2, x2)
        dec0 = self.decoder1(dec1)
        dec_end = self.decoder_end(dec0)

        pred3 = self.softmax(self.out_8(dech))
        pred2 = self.softmax(self.out_4(dec3))
        pred1 = self.softmax(self.out_2(dec2))
        pred0 = self.softmax(self.out_1(dec1))
        pred = self.softmax(self.out_end(dec_end))
        return pred, [self.up2(pred0), self.up4(pred1), self.up8(pred2), self.up16(pred3)]


# ========================================================================
# Mask / Restoration 模块 — 使用轻量参数
# ========================================================================

class MambaGenBlock(nn.Module):

    def __init__(self, dim, expansion=1, d=[0]):
        """Light: expansion 2→1"""
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.norm = nn.InstanceNorm3d(dim)
        self.down_proj = nn.Conv3d(dim, hidden_dim, kernel_size=2, stride=2)
        self.act = nn.GELU()
        self.mamba = MambaLayer_image(channels=hidden_dim, scan_modes=d, d_state=6,
                                       drop_path_rates=[0.05] * len(d))

        self.up_proj = nn.ConvTranspose3d(hidden_dim, dim, kernel_size=2, stride=2)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.down_proj(x)
        x = self.act(x)

        x = self.mamba(x)
        x = self.up_proj(x)
        return x + residual


class PromptGuidedRestorationLayer(nn.Module):
    def __init__(self, dim, spatial_size, num_modals=4, rank_ratio=8, d=[0]):
        """Light: rank_ratio 4→8"""
        super().__init__()
        self.dim = dim
        self.num_modals = num_modals

        self.modal_weight = nn.Parameter(torch.ones(num_modals))
        self.content_proj = nn.Conv3d(dim, dim, kernel_size=1)
        self.content_norm = nn.InstanceNorm3d(dim)

        self.prompt_generators = nn.ModuleList([
            SpatialChannelFactorizedToken(dim, spatial_size, rank_ratio=rank_ratio)
            for _ in range(num_modals)
        ])

        hidden_mlp = max(dim, 32)
        self.mask_mlp = nn.Sequential(
            nn.Linear(num_modals, hidden_mlp),
            nn.GELU(),
            nn.Linear(hidden_mlp, 2 * dim),
        )
        nn.init.zeros_(self.mask_mlp[-1].weight)
        nn.init.zeros_(self.mask_mlp[-1].bias)

        self.generator = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=1),
            nn.GELU(),
            MambaGenBlock(dim, expansion=1, d=d),  # Light: expansion=1
        )

        self.mse_loss = nn.MSELoss(reduction='none')
        self.cos_loss = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x, mask):
        """
        x: (B, K, C, H, W, Z)
        mask: (B, K)
        """
        B, K, C, H, W, Z = x.shape

        mask_expand = mask.view(B, K, 1, 1, 1, 1).type_as(x)

        mask_b = mask.type_as(x)
        w = F.softmax(self.modal_weight, dim=0).view(1, K).expand(B, K)
        w = w * mask_b
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        weighted = (x * w.view(B, K, 1, 1, 1, 1)).sum(dim=1)
        content = F.gelu(self.content_norm(self.content_proj(weighted)))

        mask_ctx = self.mask_mlp(mask_b)
        gamma, beta = mask_ctx.chunk(2, dim=1)
        gamma = gamma.view(B, self.dim, 1, 1, 1)
        beta = beta.view(B, self.dim, 1, 1, 1)

        out_list = []
        total_loss = torch.tensor(0.0, device=x.device).float()
        total_missing_k = 0

        for k in range(K):
            prompt = self.prompt_generators[k](B).squeeze(1)
            prompt = prompt * (1 + gamma) + beta
            gen_in = torch.cat([content, prompt], dim=1)
            fake_feat = self.generator(gen_in)

            m_k = mask_expand[:, k]
            real_feat = x[:, k]
            filled = real_feat * m_k + fake_feat * (1 - m_k)
            out_list.append(filled)

            if self.training:
                diff = self.mse_loss(fake_feat, real_feat)
                weighted_diff = diff * (1 - m_k)
                total_loss = total_loss + weighted_diff.mean()
                total_missing_k = total_missing_k + (1 - m_k).sum()

        out = torch.cat(out_list, dim=1)

        if self.training:
            if total_missing_k > 0:
                layer_loss = total_loss / total_missing_k
            else:
                layer_loss = torch.tensor(0.0, device=x.device).float()
            return out, layer_loss
        else:
            return out


class AdvancedMaskModal(nn.Module):
    def __init__(self,
                 num_modals=4,
                 dim_list=None,
                 spatial_size_list=None,
                 rank_ratio=8):
        """Light: rank_ratio 4→8"""
        super().__init__()

        if spatial_size_list is None:
            spatial_size_list = [(64, 64, 64), (32, 32, 32), (16, 16, 16), (8, 8, 8)]
        if dim_list is None:
            dim_list = [24, 48, 96, 384]

        self.num_modals = num_modals
        self.restoration_layers = nn.ModuleDict()

        d = [[0], [0], [0], [0]]
        for i, dim in enumerate(dim_list):
            spatial_size = spatial_size_list[i]
            self.restoration_layers[str(dim)] = PromptGuidedRestorationLayer(
                dim=dim,
                spatial_size=spatial_size,
                num_modals=num_modals,
                rank_ratio=rank_ratio,
                d=d[i]
            )

    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        layer = self.restoration_layers[str(C)]
        return layer(x, mask)


# ========================================================================
# Model — 编解码器解耦 + 1×1 投影
# ========================================================================

class Model(nn.Module):
    def __init__(self, num_cls=4, num_modals=4, interleaved_tokenization=False, mamba_skip=False, encoder_dims=None):
        super(Model, self).__init__()
        self.interleaved_tokenization = interleaved_tokenization
        self.num_modals = num_modals

        # ---- 编码器维度 (Lite-V2 非对称宽度) ----
        enc_dims = encoder_dims if encoder_dims is not None else globals()["encoder_dims"]

        # ---- 解码器维度 (保持 original decoder_base=12) ----
        dec_dims = [decoder_base * 2 ** i for i in range(5)]
        dec_dims[-1] = dec_dims[-1] * 2   # [12, 24, 48, 96, 384]

        self.enc_dims = enc_dims
        self.dec_dims = dec_dims

        # 编码器 (轻量)
        self.encoders = nn.ModuleList([MambaEncoder(dims=enc_dims) for _ in range(num_modals)])

        # Mask 修复模块 (仅在编码器 dims 上操作)
        self.masker = AdvancedMaskModal(
            num_modals=num_modals,
            dim_list=enc_dims[1:],  # [20, 32, 64, 192] 匹配编码器实际维度
            rank_ratio=8,
        )

        # 1×1 投影层: 编码器 dim → 解码器 dim (共享于各模态)
        # 仅 4 个投影层 (x2~x5), x1 不使用
        self.enc_to_dec_proj = nn.ModuleList([
            nn.Conv3d(enc_dims[1], dec_dims[1], 1),   # 20→24
            nn.Conv3d(enc_dims[2], dec_dims[2], 1),   # 32→48
            nn.Conv3d(enc_dims[3], dec_dims[3], 1),   # 64→96
            nn.Conv3d(enc_dims[4], dec_dims[4], 1),   # 192→384
        ])

        # 融合层 (在解码器 dims 上操作)
        self.transformer_fusion_layers = nn.ModuleList([
            ParallelDeformableFusion(dim=dec_dims[1], spatial_size=(64, 64, 64), num_modals=num_modals,
                                     rank_ratio=8, num_heads=4),
            ParallelDeformableFusion(dim=dec_dims[2], spatial_size=(32, 32, 32), num_modals=num_modals,
                                     rank_ratio=8, num_heads=4),
            ParallelDeformableFusion(dim=dec_dims[3], spatial_size=(16, 16, 16), num_modals=num_modals,
                                     rank_ratio=8, num_heads=4),
            ParallelDeformableFusion(dim=dec_dims[4], spatial_size=(8, 8, 8), num_modals=num_modals,
                                     rank_ratio=8, num_heads=4),
        ])

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)

        self.is_training = False
        self.mamba_skip = mamba_skip

        self.mse_loss = nn.MSELoss()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def _project_masker_out(self, x_flat, scale_idx):
        """
        将 masker 输出从编码器维度投影到解码器维度。
        x_flat: (B, K*enc_C, H, W, D)  →  (B, K*dec_C, H, W, D)
        """
        B, KC, H, W, D = x_flat.shape
        K = self.num_modals
        enc_C = self.enc_dims[scale_idx + 1]  # +1 因为 scale_idx 0→x2
        dec_C = self.dec_dims[scale_idx + 1]

        # 分离各模态 → 独立投影 → 拼回
        x_flat = x_flat.view(B * K, enc_C, H, W, D)
        x_flat = self.enc_to_dec_proj[scale_idx](x_flat)
        x_flat = x_flat.view(B, K * dec_C, H, W, D)
        return x_flat

    def _project_single(self, x_single, scale_idx):
        """投射单模态特征: (B, enc_C, H, W, D) → (B, dec_C, H, W, D)"""
        return self.enc_to_dec_proj[scale_idx](x_single)

    def forward(self, x, mask, return_features=False):
        encoder_outputs = []
        for i in range(self.num_modals):
            encoder_output = self.encoders[i](x[:, i:i + 1, :, :, :])
            encoder_outputs.append(encoder_output)

        x1_list, x2_list, x3_list, x4_list, x5_list = zip(*encoder_outputs)

        sep_preds = []
        if self.training:
            # --- 单模态分支：投影后送 decoder_sep ---
            for i in range(self.num_modals):
                x2_p = self._project_single(x2_list[i], 0)
                x3_p = self._project_single(x3_list[i], 1)
                x4_p = self._project_single(x4_list[i], 2)
                x5_p = self._project_single(x5_list[i], 3)
                sep_pred = self.decoder_sep(None, x2_p, x3_p, x4_p, x5_p)
                sep_preds.append(sep_pred)

            # --- 融合分支：masker(enc dims) → 投影(dec dims) → fusion → decoder ---
            sim_loss = 0.0

            s_x2, sim_loss_2 = self.masker(torch.stack(x2_list, dim=1), mask)
            s_x3, sim_loss_3 = self.masker(torch.stack(x3_list, dim=1), mask)
            s_x4, sim_loss_4 = self.masker(torch.stack(x4_list, dim=1), mask)
            s_x5, sim_loss_5 = self.masker(torch.stack(x5_list, dim=1), mask)
            sim_loss += sim_loss_2 + sim_loss_3 + sim_loss_4 + sim_loss_5
            sim_loss = sim_loss / 4.0

            s_z2 = self._project_masker_out(s_x2, 0)
            s_z3 = self._project_masker_out(s_x3, 1)
            s_z4 = self._project_masker_out(s_x4, 2)
            s_z5 = self._project_masker_out(s_x5, 3)

            s_z2 = self.transformer_fusion_layers[0](s_z2, mask)
            s_z3 = self.transformer_fusion_layers[1](s_z3, mask)
            s_z4 = self.transformer_fusion_layers[2](s_z4, mask)
            s_z5 = self.transformer_fusion_layers[3](s_z5, mask)

            fuse_pred, preds = self.decoder_fuse(None, s_z2, s_z3, s_z4, s_z5)

            if return_features:
                feature_dict = {
                    'raw_enc': [torch.stack(x2_list, dim=1), torch.stack(x3_list, dim=1),
                                torch.stack(x4_list, dim=1), torch.stack(x5_list, dim=1)],
                    'restored_enc': [s_x2, s_x3, s_x4, s_x5],
                    'fused_dec': [s_z2, s_z3, s_z4, s_z5],
                }
                return fuse_pred, sep_preds, preds, sim_loss, feature_dict
            return fuse_pred, sep_preds, preds, sim_loss
        else:
            # --- 推理路径 ---
            x2 = self.masker(torch.stack(x2_list, dim=1), mask)
            x3 = self.masker(torch.stack(x3_list, dim=1), mask)
            x4 = self.masker(torch.stack(x4_list, dim=1), mask)
            x5 = self.masker(torch.stack(x5_list, dim=1), mask)

            x2 = self._project_masker_out(x2, 0)
            x3 = self._project_masker_out(x3, 1)
            x4 = self._project_masker_out(x4, 2)
            x5 = self._project_masker_out(x5, 3)

            x2 = self.transformer_fusion_layers[0](x2, mask)
            x3 = self.transformer_fusion_layers[1](x3, mask)
            x4 = self.transformer_fusion_layers[2](x4, mask)
            x5 = self.transformer_fusion_layers[3](x5, mask)
            fuse_pred, preds = self.decoder_fuse(None, x2, x3, x4, x5)
            if return_features:
                feature_dict = {
                    'raw_enc': [torch.stack(x2_list, dim=1), torch.stack(x3_list, dim=1),
                                torch.stack(x4_list, dim=1), torch.stack(x5_list, dim=1)],
                    'restored_enc': [x2, x3, x4, x5],
                    'fused_dec': [x2, x3, x4, x5],
                }
                return fuse_pred, feature_dict
            return fuse_pred
