import torch.nn as nn
import torch

'''
The Pipeline: 
Input image
    → Stem (initial conv)
    → MBConv blocks (feature extraction)
    → Head (final conv)
    → Global pooling
    → Dropout
    → Fully connected layer (classification)
'''

'''
EfficientNet-B0 — adapted from MobileNetV3-Small (DS 6050)

Changes from MobileNetV3-Small to EfficientNet-B0:

1. Replace Hard-Swish/Hard-Sigmoid with Swish (SiLU)
   EfficientNet uses the smooth Swish activation (x * sigmoid(x)) everywhere,
   implemented as nn.SiLU(). No hard approximations needed — modern hardware
   handles it efficiently. HardSigmoid/HardSwish are removed.

2. Change SE reduction ratio from 1/4 (of in_channels) to 1/4 (of hidden_dim)
   MobileNetV3 reduces relative to the block's input channels.
   EfficientNet reduces relative to the *expanded* (hidden) channels and
   uses a fixed ratio of 0.25, so: se_reduced = max(1, int(hidden_dim * 0.25)).

3. Add a width multiplier (alpha) and depth multiplier (phi)
   EfficientNet-B0 is the baseline (phi=1.0). B0 uses:
     width_mult = 1.0, depth_mult = 1.0, resolution = 224
   The depth multiplier scales *how many times* each block is repeated:
     repeats = max(1, int(math.ceil(base_repeats * depth_mult)))
   The block configuration now includes a `repeats` column.

4. Change the block configuration to EfficientNet-B0's 7-stage table
   MobileNetV3-Small has 11 blocks with non-uniform expand ratios (4.5, 3.67…).
   EfficientNet-B0 uses 7 stages, uniform expand ratio of 6 (except stage 1 = 1),
   and alternating 3×3 / 5×5 kernels. All stages use SE. All use Swish.

5. Change the stem convolution from 16 channels to 32 channels
   MobileNetV3-Small's initial conv outputs 16 channels.
   EfficientNet-B0 starts at 32 channels (stride 2).

6. Change the head: final conv expands to 1280 channels (not 576 → 1024)
   EfficientNet-B0's head: Conv1×1(last_ch → 1280) → Swish → AvgPool
   → Dropout → FC(num_classes). The pooling moves before the classifier FC.
'''


# Swish activation: x * sigmoid(x)
# nn.SiLU is PyTorch's built-in Swish — numerically stable and fast.
class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.silu = nn.SiLU(inplace=inplace)

    # apply swish to input
    def forward(self, x):
        return self.silu(x)


### channel attention mechanism ###
# SE block — same structure as MobileNetV3 but reduction is over hidden_dim.
# The paper uses se_ratio = 0.25 applied to the *expanded* channel count.
class SqueezeExcitation(nn.Module):
    def __init__(self, hidden_dim, se_ratio=0.25):
        super(SqueezeExcitation, self).__init__()
        # compute bottleneck size
        reduced_dim = max(1, int(hidden_dim * se_ratio))

        # AdaptiveAvgPool2d(1) is global avg pool [B, C, H, W] → [B, C, 1, 1]
        # nn.Conv2d(hidden_dim,... and nn.SiLU(inplace=True),
        #   are 1st FC layer (via 1×1 conv) → Sigmoid
        # Sigmoid (not hard-sigmoid) per the original EfficientNet paper.
        # nn.Conv2d(reduced_dim,... and nn.Sigmoid
        #   expand back to original channels, weights between 0 and 1
        self.squeeze_excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, reduced_dim, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_dim, hidden_dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    # Multiply attention weights with input
    def forward(self, x):
        return x * self.squeeze_excitation(x)


# MBConv block — EfficientNet's version of the inverted residual.

# Structure:
# Input
# → (Optional) Expansion 1×1 conv
# → Depthwise conv
# → SE block
# → Projection 1×1 conv
# → (Optional) Residual connection

# Every block uses Swish and SE (unlike V3-Small where some blocks skip SE).
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, expand_ratio=1, se_ratio=0.25):
        super(MBConv, self).__init__()

        # use skip connection if shape doesn't change
        self.residual_will_be_used = stride == 1 and in_channels == out_channels

        # expansion size
        hidden_dim = int(in_channels * expand_ratio)

        list_of_layers = []

        # Expansion phase (pointwise, only when expand_ratio > 1)
        # Stage 1 uses expand_ratio=1 so this branch is skipped there.
        # 1×1 conv expands channels
        if expand_ratio != 1:
            list_of_layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish(inplace=True)
            ])

        # Depthwise convolution (groups = hidden_dim)
        # in_channels=hiddden_dim; increases channels (e.g., 32 -> 192)
        # groups=hidden_dim; each channel convolved independently 
        list_of_layers.extend([
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=hidden_dim,
                bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            Swish(inplace=True)
        ])

        # Squeeze-and-Excitation (always applied in EfficientNet)
        list_of_layers.append(SqueezeExcitation(hidden_dim, se_ratio))

        # Projection phase 
        #   reduces channels back down
        #   linear bottleneck — no activation
        list_of_layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        # wrap layers
        self.sequential = nn.Sequential(*list_of_layers)

    # skip connection, else no skip
    def forward(self, x):
        if self.residual_will_be_used:
            return x + self.sequential(x)
        else:
            return self.sequential(x)


class EfficientNet(nn.Module):
    '''
    EfficientNet-B0 (phi=1.0).

    Block config columns:
        (in_c, out_c, kernel, stride, expand_ratio, repeats)

    All blocks use SE with se_ratio=0.25 and Swish activation.
    Repeats are scaled by depth_mult; channels scaled by width_mult.
    '''

    def __init__(self, num_classes=18, width_mult=1.0,
                 depth_mult=1.0, dropout_prob=0.2):
        super(EfficientNet, self).__init__()

        import math

        def scaled_channels(ch):
            '''Round to nearest multiple of 8 after width scaling.'''
            new_ch = ch * width_mult
            # Round to the nearest multiple of 8, at least 1
            new_ch = max(8, int(new_ch + 4) // 8 * 8)
            return new_ch

        # Scale depth
        def scaled_repeats(reps):
            return max(1, int(math.ceil(reps * depth_mult)))

        # ── Stem convolution (stride 2, 3×3) ──────────────────────────────
        stem_channels = scaled_channels(32)

        # Input RGB image
        # Downsample (stride=2)
        self.initial_convolution = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(stem_channels),
            Swish(inplace=True)
        )

        # ── EfficientNet-B0 stage configuration ───────────────────────────
        # Each tuple: (in_c, out_c, kernel, stride, expand_ratio, repeats)
        configuration = [
            (32,  16,  3, 1, 1, 1),   # Stage 1 — no expansion, 3×3
            (16,  24,  3, 2, 6, 2),   # Stage 2
            (24,  40,  5, 2, 6, 2),   # Stage 3 — 5×5 kernel
            (40,  80,  3, 2, 6, 3),   # Stage 4
            (80,  112, 5, 1, 6, 3),   # Stage 5 — 5×5 kernel
            (112, 192, 5, 2, 6, 4),   # Stage 6 — 5×5 kernel
            (192, 320, 3, 1, 6, 1),   # Stage 7
        ]

        list_of_layers = []

        # Build all blocks across all stages.
        # The first block in each stage may have stride > 1 (downsampling);
        # all subsequent repeated blocks in that stage use stride = 1.
        # Loop through stages
        for in_c, out_c, kernel, stride, expand_ratio, repeats in configuration:
            scaled_in  = scaled_channels(in_c)
            scaled_out = scaled_channels(out_c)
            num_repeats = scaled_repeats(repeats)

            for i in range(num_repeats):
                # Only the first block uses the stage's stride and in_channels.
                # Repeated blocks connect scaled_out → scaled_out at stride 1.
                block_stride   = stride if i == 0 else 1
                block_in       = scaled_in if i == 0 else scaled_out

                # wrap all blocks
                list_of_layers.append(
                    MBConv(
                        in_channels=block_in,
                        out_channels=scaled_out,
                        kernel_size=kernel,
                        stride=block_stride,
                        expand_ratio=expand_ratio,
                        se_ratio=0.25
                    )
                )

        self.sequential = nn.Sequential(*list_of_layers)

        # ── Head ──────────────────────────────────────────────────────────
        # Conv 1×1 expands to 1280 features before pooling.
        last_channels = scaled_channels(320)

        if width_mult > 1.0:
            head_channels = scaled_channels(1280)
        else:
            head_channels = 1280

        self.final_convolution = nn.Sequential(
            nn.Conv2d(last_channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            Swish(inplace=True)
        )

        # Global average pooling: [B, C, H, W] → [B, C, 1, 1]
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # dropout
        self.dropout = nn.Dropout(dropout_prob)

        # Classifier
        # Single linear layer — no intermediate FC head like V3-Small
        self.linear_transformation = nn.Linear(head_channels, num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        '''
        Conv: kaiming_normal_ (fan_out mode, nonlinearity='relu')
        BatchNorm: weight=1, bias=0
        Linear: normal(0, 0.01) — tighter init than V3 for stability
        '''
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # x
        # ↓
        # initial_convolution (stem 3×3 s=2, 32ch)
        # ↓
        # MBConv stages (S1–S7, 16 total blocks at B0)
        # ↓
        # final_convolution (1×1 → 1280ch)
        # ↓
        # average_pooling → [B, 1280, 1, 1]
        # ↓
        # flatten → [B, 1280]
        # ↓
        # dropout
        # ↓
        # linear_transformation → [B, num_classes]
        intermediate = self.initial_convolution(x)
        intermediate = self.sequential(intermediate)
        intermediate = self.final_convolution(intermediate)
        intermediate = self.average_pooling(intermediate)
        intermediate = torch.flatten(intermediate, start_dim=1)
        intermediate = self.dropout(intermediate)
        return self.linear_transformation(intermediate)
