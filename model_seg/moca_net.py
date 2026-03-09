import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch.cuda.amp import autocast
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from torch.utils.checkpoint import checkpoint


basic_dims = 12
transformer_basic_dims = 128
mlp_dim = 4096
num_heads = 8
depth = 1
patch_size = 8
input_patch_size = 128


class Softmax_32(nn.Module):
    def __init__(self):
        super(Softmax_32, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        return self.softmax(x)


class BidirectionalMambaBlocks(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        d_inner = d_model // 2
        self.mamba_fwd = Mamba(d_model=d_inner, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_inner, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        residual = x
        x = self.norm(x)
        x_fwd, x_bwd = torch.split(x, self.d_model // 2, dim=-1)
        out_fwd = self.mamba_fwd(x_fwd)
        out_bwd = self.mamba_bwd(x_bwd.flip(dims=[1])).flip(dims=[1])
        out = torch.cat([out_fwd, out_bwd], dim=-1)
        return out + residual


class MambaLayer_image(nn.Module):
    def __init__(self, channels, scan_modes=None, d_state=8):
        super().__init__()
        self.channels = channels
        if scan_modes is None:
            scan_modes = [0, 1, 2]

        # 映射扫描模式
        self.scan_modes = []
        for mode in scan_modes:
            if isinstance(mode, str):
                mode_map = {'DHW': 0, 'HWD': 1, 'WDH': 2}
                self.scan_modes.append(mode_map[mode.upper()])
            else:
                self.scan_modes.append(mode)
        self.depth = len(self.scan_modes)
        self.layers = nn.ModuleList([
            BidirectionalMambaBlocks(d_model=channels, d_state=d_state)
            for _ in range(self.depth)
        ])

    def forward(self, x):
        B, C, D, H, W = x.shape
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
            else:  # WDH
                x_permuted = current_feat.permute(0, 1, 4, 2, 3)
                feat_size = (W, D, H)
                inverse_permute = (0, 1, 3, 4, 2)

            x_flat = x_permuted.reshape(B, C, -1).transpose(-1, -2)
            x_mamba = layer(x_flat)
            x_out = x_mamba.transpose(-1, -2).reshape(B, C, *feat_size)

            if inverse_permute is not None:
                current_feat = x_out.permute(*inverse_permute)
            else:
                current_feat = x_out
        return current_feat + x


class SpatialChannelFactorizedToken(nn.Module):
    def __init__(self, dim, spatial_size, rank_ratio=4):
        super().__init__()
        self.rank = max(1, dim // rank_ratio)
        self.spatial_size = spatial_size
        self.spatial_basis = nn.Parameter(torch.randn(1, self.rank, *spatial_size))
        self.mixer = nn.Conv3d(self.rank, dim, kernel_size=1, bias=False)
        nn.init.normal_(self.spatial_basis, std=0.02)
        nn.init.normal_(self.mixer.weight, std=0.02)

    def forward(self, B):
        token = self.mixer(self.spatial_basis)  # (1, Dim, H, W, D)
        token = token.unsqueeze(1)  # (1, 1, Dim, H, W, D)
        token = token.expand(B, -1, -1, *self.spatial_size)
        return token


class DeformableAlignmentBlock(nn.Module):
    """
    独立的可变形对齐模块。
    输入: [B, M, C, H, W, D]
    输出: [B, M, C, H, W, D] (已对齐)
    """

    def __init__(self, in_channels, num_modals, max_displacement=1.0 / 32.0):
        super().__init__()
        self.max_displacement = max_displacement

        # 偏移生成网络
        self.offset_conv = nn.Sequential(
            nn.Conv3d(in_channels * num_modals, num_modals * 4, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(num_modals * 4),
            nn.GELU(),
            nn.Conv3d(num_modals * 4, num_modals * 3, kernel_size=3, padding=1, bias=True),
        )

        # 零初始化最后一层，确保初始状态下不做偏移
        nn.init.constant_(self.offset_conv[-1].weight, 0)
        nn.init.constant_(self.offset_conv[-1].bias, 0)

    def forward(self, x):
        B, M, C, H, W, D = x.shape
        # 拼接所有模态以计算偏移
        x_concat = x.view(B, M * C, H, W, D)

        # 计算 offsets: [B, M*3, H, W, D]
        offsets = torch.tanh(self.offset_conv(x_concat)) * self.max_displacement
        offsets = offsets.view(B * M, 3, H, W, D)

        # 生成基础网格并采样
        base_grid = self._generate_grid(D, H, W, x.device).unsqueeze(0).expand(B * M, -1, -1, -1, -1)
        # grid shape: [B*M, D, H, W, 3] -> offsets permute to match
        sampling_grid = base_grid + offsets.permute(0, 2, 3, 4, 1)

        x_reshaped = x.view(B * M, C, H, W, D)
        # align_corners=True 对应 generate_grid 的 linspace(-1, 1)
        x_aligned = F.grid_sample(x_reshaped, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)

        return x_aligned.view(B, M, C, H, W, D)

    def _generate_grid(self, D, H, W, device):
        # 这里的顺序必须对应 grid_sample 的期望 (x, y, z) 即 (w, h, d)
        d = torch.linspace(-1, 1, D, device=device)
        h = torch.linspace(-1, 1, H, device=device)
        w = torch.linspace(-1, 1, W, device=device)
        mesh = torch.meshgrid(d, h, w, indexing='ij')  # D, H, W
        # grid_sample expects coordinates in range [-1, 1] for the D, H, W dimensions
        # The last dimension should be (x, y, z) -> (W, H, D)
        return torch.stack([mesh[2], mesh[1], mesh[0]], dim=-1)


class ParallelDeformableFusion(nn.Module):
    """
    仅负责融合的模块。
    输入: [B, M, C, H, W, D] (预期已对齐)
    输出: [B, C, H, W, D] (融合后的特征)
    """

    def __init__(self, dim, spatial_size, num_modals=4, rank_ratio=4):
        super().__init__()
        self.dim = dim
        self.num_modals = num_modals
        self.total_units = num_modals + 1  # M 个模态 + 1 个 Fused Token

        # Token 生成器
        self.token_generator = SpatialChannelFactorizedToken(dim, spatial_size, rank_ratio)

        # 动态调整膨胀率
        s_size = spatial_size[0] if isinstance(spatial_size, (tuple, list)) else spatial_size
        k, d = 3, 1  # Default
        if s_size >= 32: k, d = 5, 1

        # Attention Branch: Q, K, V
        self.to_q = nn.Conv3d(self.total_units * dim, self.total_units * dim, 1, groups=self.total_units, bias=False)
        self.to_k = nn.Conv3d(self.total_units * dim, self.total_units * dim, 1, groups=self.total_units, bias=False)
        self.to_v = nn.Conv3d(self.total_units * dim, self.total_units * dim, 1, groups=self.total_units, bias=False)

        # CNN Branch
        p = (k - 1) * d // 2
        self.cnn = nn.Sequential(
            nn.Conv3d(dim * num_modals, dim * 2, kernel_size=k, padding=p, dilation=d),
            nn.GELU(),
            nn.Conv3d(dim * 2, dim, kernel_size=k, padding=p, dilation=d)
        )

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x_aligned):
        """
        x_aligned: [B, M, C, H, W, D]
        """
        B, M, C, H, W, D = x_aligned.shape

        # 1. 生成可学习的 Fused Token
        fused_token = self.token_generator(B)  # [B, 1, C, H, W, D]

        # 2. 拼接: (B, M+1, C, H, W, D)
        x_all = torch.cat([x_aligned, fused_token], dim=1)
        x_grouped_in = x_all.view(B, (M + 1) * C, H, W, D)

        # 3. 计算 Q, K, V
        q = self.to_q(x_grouped_in).view(B, M + 1, C, H, W, D)
        k = self.to_k(x_grouped_in).view(B, M + 1, C, H, W, D)
        v = self.to_v(x_grouped_in).view(B, M + 1, C, H, W, D)

        # 4. Attention
        scale = C ** -0.5
        attn = torch.einsum('bichwd, bjchwd -> bijhwd', q, k) * scale
        attn = attn.softmax(dim=2)  # Softmax over Key dimension

        # 5. Aggregate Values
        out = torch.einsum('bijhwd, bjchwd -> bichwd', attn, v)  # [B, M+1, C, H, W, D]

        # 6. 提取 Fused Token 对应的输出 (最后一个)
        att_fused_out = out[:, -1, ...]  # [B, C, H, W, D]

        # 7. CNN Branch 融合
        conv_input = x_aligned.view(B, M * C, H, W, D)
        cnn_fused_out = self.cnn(conv_input)

        # 8. 最终融合
        return cnn_fused_out + self.alpha * att_fused_out


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(hidden_size, mlp_dim, 1),
            nn.GELU(),
            nn.Conv3d(mlp_dim, hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)


class GSC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, 1, 1),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, 3, 1, 1),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1, 1, 0),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1, 1, 0),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return self.final(x1 + x2) + x


class StemDWResidual(nn.Module):
    def __init__(self, in_channels=1, out_channels=basic_dims):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.block = nn.Sequential(
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=5, padding=2, groups=out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
        )
        self.out_norm = nn.GroupNorm(4, out_channels)
        self.out_act = nn.GELU()

    def forward(self, x):
        y = self.proj(x)
        y = y + self.block(y)
        return self.out_act(self.out_norm(y))


class MambaEncoder(nn.Module):
    def __init__(self, scan_modes=[[], [0], [0], [0], [0, 0, 0]], dims=None):
        super().__init__()
        if dims is None:
            dims = [basic_dims * 2 ** i for i in range(5)]
            dims[-1] = dims[-1] * 2

        self.stem = StemDWResidual()
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.norm = nn.ModuleList()

        # Level 0
        self.norm.append(nn.InstanceNorm3d(dims[0]))
        self.mlps.append(MlpChannel(dims[0], 2 * dims[0]))

        # Levels 1-4
        for i in range(1, 5):
            self.downsample_layers.append(nn.Sequential(
                nn.InstanceNorm3d(dims[i - 1]),
                nn.Conv3d(dims[i - 1], dims[i], kernel_size=2, stride=2),
            ))
            self.stages.append(MambaLayer_image(channels=dims[i], scan_modes=scan_modes[i]))
            self.gscs.append(GSC(dims[i]))
            self.norm.append(nn.InstanceNorm3d(dims[i]))
            self.mlps.append(MlpChannel(dims[i], 2 * dims[i]))

    def forward(self, x):
        outs = []
        with autocast(enabled=True):
            x = self.stem(x)
            x_out = self.mlps[0](self.norm[0](x))
            x, x_out = x.float(), x_out.float()
            outs.append(x_out)

            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.gscs[i](x)
                x = self.stages[i](x)
                x_out = self.mlps[i + 1](self.norm[i + 1](x))
                outs.append(x_out)
        return tuple(outs)


# Decoders (Keep as is, just compacted for brevity)
class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4):
        super().__init__()
        self.decoderh = UnetrBasicBlock(3, basic_dims * 32, basic_dims * 16, 3, 1, "instance", True)
        self.decoder4 = UnetrUpBlock(3, basic_dims * 16, basic_dims * 8, 3, 2, "instance", True)
        self.decoder3 = UnetrUpBlock(3, basic_dims * 8, basic_dims * 4, 3, 2, "instance", True)
        self.decoder2 = UnetrUpBlock(3, basic_dims * 4, basic_dims * 2, 3, 2, "instance", True)
        self.decoder1 = nn.ConvTranspose3d(basic_dims * 2, basic_dims * 2, 2, 2)
        self.out_end = UnetOutBlock(3, basic_dims * 2, num_cls)
        self.softmax = Softmax_32()

    def forward(self, _, x2, x3, x4, x5):
        d = self.decoderh(x5)
        d = self.decoder4(d, x4)
        d = self.decoder3(d, x3)
        d = self.decoder2(d, x2)
        d = self.decoder1(d)
        return self.softmax(self.out_end(d))


class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=4):
        super().__init__()
        self.decoderh = UnetrBasicBlock(3, basic_dims * 32, basic_dims * 16, 3, 1, "instance", True)
        self.decoder4 = UnetrUpBlock(3, basic_dims * 16, basic_dims * 8, 3, 2, "instance", True)
        self.decoder3 = UnetrUpBlock(3, basic_dims * 8, basic_dims * 4, 3, 2, "instance", True)
        self.decoder2 = UnetrUpBlock(3, basic_dims * 4, basic_dims * 2, 3, 2, "instance", True)
        self.decoder1 = nn.ConvTranspose3d(basic_dims * 2, basic_dims * 2, 2, 2)
        self.decoder_end = UnetrBasicBlock(3, basic_dims * 2, basic_dims * 2, 3, 1, "instance", True)

        # Deep supervision heads
        self.out_8 = UnetOutBlock(3, basic_dims * 16, num_cls)
        self.out_4 = UnetOutBlock(3, basic_dims * 8, num_cls)
        self.out_2 = UnetOutBlock(3, basic_dims * 4, num_cls)
        self.out_1 = UnetOutBlock(3, basic_dims * 2, num_cls)
        self.out_end = UnetOutBlock(3, basic_dims * 2, num_cls)
        self.softmax = Softmax_32()

        self.ups = nn.ModuleList([
            nn.Upsample(scale_factor=s, mode='trilinear', align_corners=True)
            for s in [2, 4, 8, 16]
        ])

    def forward(self, _, x2, x3, x4, x5):
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

        return pred, [self.ups[0](pred0), self.ups[1](pred1), self.ups[2](pred2), self.ups[3](pred3)]


class MambaGenBlock(nn.Module):
    def __init__(self, dim, expansion=2, d=[0]):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.norm = nn.InstanceNorm3d(dim)
        self.down_proj = nn.Conv3d(dim, hidden_dim, 2, 2)
        self.act = nn.GELU()
        self.mamba = MambaLayer_image(channels=hidden_dim, scan_modes=d)
        self.up_proj = nn.ConvTranspose3d(hidden_dim, dim, 2, 2)

    def forward(self, x):
        res = x
        x = self.act(self.down_proj(self.norm(x)))
        x = self.up_proj(self.mamba(x))
        return x + res


class PromptGuidedRestorationLayer(nn.Module):
    def __init__(self, dim, spatial_size, num_modals=4, rank_ratio=4, d=[0]):
        super().__init__()
        self.content_fusion = nn.Sequential(
            nn.Conv3d(dim * num_modals, dim, 1, groups=dim),
            nn.InstanceNorm3d(dim),
            nn.GELU()
        )
        self.prompt_generators = nn.ModuleList([
            SpatialChannelFactorizedToken(dim, spatial_size, rank_ratio) for _ in range(num_modals)
        ])
        self.generator = nn.Sequential(
            nn.Conv3d(dim * 2, dim, 1),
            nn.GELU(),
            MambaGenBlock(dim, expansion=2, d=d),
        )
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, x, mask):
        # x: [B, K, C, H, W, Z]
        B, K, C, H, W, Z = x.shape
        mask_expand = mask.view(B, K, 1, 1, 1, 1).type_as(x)

        # 提取上下文信息
        feat = (x * mask_expand).permute(0, 2, 1, 3, 4, 5).reshape(B, C * K, H, W, Z)
        content = self.content_fusion(feat)

        out_list = []
        total_loss = torch.tensor(0.0, device=x.device)
        total_missing = 0.0

        for k in range(K):
            prompt = self.prompt_generators[k](B).squeeze(1)
            fake_feat = self.generator(torch.cat([content, prompt], dim=1))

            m_k = mask_expand[:, k]
            real_feat = x[:, k]

            # 如果是 mask 掉的区域用 fake，否则用 real
            filled = real_feat * m_k + fake_feat * (1 - m_k)
            out_list.append(filled)

            if self.training:
                # 仅计算缺失区域的 Loss
                loss_map = self.mse_loss(fake_feat, real_feat) * (1 - m_k)
                total_loss += loss_map.mean()
                total_missing += (1 - m_k).sum()

        out = torch.cat(out_list, dim=1)  # [B, K*C, ...] -> 这里的 reshape 在外部处理比较好，这里先返回 concat 的

        if self.training:
            layer_loss = total_loss / (total_missing + 1e-6) if total_missing > 0 else total_loss
            return out, layer_loss
        return out


class AdvancedMaskModal(nn.Module):
    def __init__(self, num_modals=4, dim_list=None, spatial_size_list=None):
        super().__init__()
        if spatial_size_list is None:
            spatial_size_list = [(64, 64, 64), (32, 32, 32), (16, 16, 16), (8, 8, 8)]
        if dim_list is None:
            dim_list = [24, 48, 96, 384]  # 对应 x2, x3, x4, x5 的通道数

        self.restoration_layers = nn.ModuleDict()
        d = [[0], [0], [0], [0]]
        for i, dim in enumerate(dim_list):
            self.restoration_layers[str(dim)] = PromptGuidedRestorationLayer(
                dim=dim,
                spatial_size=spatial_size_list[i],
                num_modals=num_modals,
                d=d[i]
            )

    def forward(self, x, mask):
        # x: [B, K, C, H, W, Z]
        B, K, C, H, W, Z = x.size()
        return self.restoration_layers[str(C)](x, mask)


# -----------------------------
# 主网络 (Moca_net)
# -----------------------------

class Moca_net(nn.Module):
    def __init__(self, num_cls=4, num_modals=4, mamba_skip=False):
        super(Moca_net, self).__init__()
        self.num_modals = num_modals
        self.encoders = nn.ModuleList([MambaEncoder() for _ in range(num_modals)])
        self.masker = AdvancedMaskModal(num_modals=num_modals)


        # 注意：Encoder 输出顺序是 x1, x2, x3, x4, x5
        # 融合通常在深层特征上进行：x2(64^3), x3(32^3), x4(16^3), x5(8^3)
        # dims: x2->24, x3->48, x4->96, x5->384 (假设 basic_dims=12)

        self.fusion_dims = [basic_dims * 2, basic_dims * 4, basic_dims * 8, basic_dims * 32]
        self.fusion_sizes = [(64, 64, 64), (32, 32, 32), (16, 16, 16), (8, 8, 8)]

        # 分离的对齐模块
        self.align_modules = nn.ModuleList([
            DeformableAlignmentBlock(dim, num_modals)
            for dim in self.fusion_dims
        ])

        # 分离的融合模块
        self.fusion_modules = nn.ModuleList([
            ParallelDeformableFusion(dim, size, num_modals)
            for dim, size in zip(self.fusion_dims, self.fusion_sizes)
        ])

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)

        self.training = True  # Default

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x, mask):
        # 1. 独立编码 (Encoders)
        encoder_results = []
        for i in range(self.num_modals):
            encoder_results.append(self.encoders[i](x[:, i:i + 1, ...]))

        # Indices: 0->x1(full res), 1->x2, 2->x3, 3->x4, 4->x5
        x_lists = list(zip(*encoder_results))

        sep_preds = []
        if self.training:
            for i in range(self.num_modals):
                # 传入 x2...x5 的单个模态特征
                p = self.decoder_sep(None, x_lists[1][i], x_lists[2][i], x_lists[3][i], x_lists[4][i])
                sep_preds.append(p)

        # 输入: Stacked [B, M, C, H, W, D]
        restored_feats = []
        sim_loss_total = 0.0

        # indices 1 to 4 correspond to features x2 to x5
        for i in range(1, 5):
            feat_stack = torch.stack(x_lists[i], dim=1)  # [B, M, C, D, H, W]

            if self.training:
                feat_restored, loss = self.masker(feat_stack, mask)
                sim_loss_total += loss
            else:
                # 推理时也经过 Masker 填补缺失模态
                feat_restored = self.masker(feat_stack, mask)

            # [B, M, C, D, H, W]
            B, _, D, H, W = feat_restored.shape
            M = self.num_modals
            C = self.fusion_dims[i - 1]
            feat_restored = feat_restored.view(B, M, C, D, H, W)

            restored_feats.append(feat_restored)

        sim_loss_total /= 4.0

        fused_feats = []
        for i in range(4):  # 对应 x2, x3, x4, x5
            feat = restored_feats[i]  # [B, M, C, D, H, W]
            # Step A: Deformable Alignment
            feat_aligned = self.align_modules[i](feat)
            # Step B: Parallel Deformable Fusion
            feat_fused = self.fusion_modules[i](feat_aligned)
            fused_feats.append(feat_fused)

        # Decoder inputs: None, x2_fused, x3_fused, x4_fused, x5_fused
        fuse_pred, deep_preds = self.decoder_fuse(None, *fused_feats)

        if self.training:
            return fuse_pred, sep_preds, deep_preds, sim_loss_total
        else:
            return fuse_pred
