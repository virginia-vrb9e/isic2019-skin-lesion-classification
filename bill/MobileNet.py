import torch.nn as nn
import torch

'''
Changes from (DS 6050 Assignment 2) MobileNet to MobileNetV3-Small

1. Replace ReLU6 with Hard-Swish (and Hard-Sigmoid)
MobileNetV3 replaces many ReLU6 activations with hard-swish because it improves accuracy with little cost.

New activation modules — HardSigmoid and HardSwish V3 replaces ReLU6 with Hard Swish (in later layers) and Hard Sigmoid (inside SE blocks). 
Both are implemented as small nn.Module subclasses built from ReLU6 so they stay hardware-friendly.

2. Add Squeeze-and-Excitation (SE) Blocks
MobileNetV3 adds channel attention using SE modules inside inverted residual blocks.

New SqueezeExcitation block V3-Small adds SE blocks to most bottlenecks. 
It global-average-pools the feature map, passes it through two 1×1 convolutions 
(ReLU → HardSigmoid), and multiplies the result back as a channel-wise attention gate.

3. Change the Inverted Residual Block
InvertedResidual extended with kernel_size, use_se, and use_hs 
The V3 bottlenecks use 5×5 depthwise kernels in the later stages (vs always 3×3 in V2), 
optionally insert a SE block after the depthwise step, 
and choose between ReLU and Hard Swish per-block. 
The residual logic is unchanged.

4. Allow 5×5 Depthwise Convolutions
MobileNetV3 uses kernel size 5 in many blocks.

5. Change the Architecture Configuration
MobileNetV3-Small uses a different block configuration.
MobileNet backbone reconfigured for V3-Small The 11-block configuration table matches the paper exactly 
— narrower initial conv (16 ch instead of 32),
- non-uniform expand ratios (4.5, 3.67, etc.),
- and 5×5 kernels in the SE+HS blocks.

6. Add the MobileNetV3 Head
The final layers differ.
Updated final conv and classifier head V3-Small's head is 
pool → Conv1×1(576→1024) → HardSwish → Dropout → FC, 
replacing V2's simpler pool → flatten → dropout → FC. 
The public API (average_pooling, dropout, linear_transformation) is preserved.
'''
# A cheap approximation of the sigmoid function used in MobileNetV3
# True sigmoid: σ(x)=1/(1+e^(-x) )
# expensive due to the exponential

# HardSigmoid approximation: relu6(x + 3) / 6
class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    # roughly behaves like sigmoid but cheaper
    def forward(self, x):
        return self.relu6(x + 3.0) / 6.0

# Efficient approximation of Swish activation
# from Swish(x)=x⋅sigmoid(x) to x⋅HardSigmoid(x)
class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.hard_sigmoid = HardSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.hard_sigmoid(x)

# SE attention mechanism from the paper: "Squeeze-and-Excitation Networks"
# Let the network learn which channels are important
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()

        # Squeeze: global average pool then two FC layers with ReLU and HardSigmoid
        # Feature map [B, C, H, W]
        # Global average pooling: [B, C, H, W] → [B, C, 1, 1]
        #   Each channel becomes one number
        # Two 1×1 convolutions act like fully connected layers: Conv2d...

        self.squeeze_excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, in_channels, kernel_size=1, bias=True),
            HardSigmoid(inplace=True)
        )
    # Channels get multiplied by learned weights.
    def forward(self, x):
        return x * self.squeeze_excitation(x)


# Core block of MobileNetV2/V3. 
# The name comes from inverting the usual bottleneck structure.
# Traditional CNN bottleneck: wide → narrow → wide
# Inverted bottleneck: narrow → wide → narrow (saves computation)

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 expand_ratio=1, use_se=False, use_hs=False):
        super(InvertedResidual, self).__init__()

        # Residual connection is used when stride is 1 and channels match
        self.residual_will_be_used = stride == 1 and in_channels == out_channels

        hidden_dim = int(in_channels * expand_ratio)
        activation = HardSwish(inplace=True) if use_hs else nn.ReLU(inplace=True)

        list_of_layers = []

        # Expansion phase (pointwise, only if expand_ratio != 1)
        if expand_ratio != 1:
            list_of_layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation if expand_ratio != 1 else nn.Identity()
            ])
            # Re-create activation for depthwise (avoid reusing same inplace module)
            activation = HardSwish(inplace=True) if use_hs else nn.ReLU(inplace=True)

        # Depthwise convolution
        # groups = hidden_dim means each channel has its own convolution filter.
        # Normal conv cost: k^2*C_in*C_out
        # Depthwise cost: k^2*C
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
            activation
        ])

        # Squeeze-and-Excitation - adds channel attention
        if use_se:
            reduced_dim = max(1, in_channels // 4)
            list_of_layers.append(SqueezeExcitation(hidden_dim, reduced_dim))

        # Projection phase (linear bottleneck, no activation)
        list_of_layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.sequential = nn.Sequential(*list_of_layers)

    # Residual is used only when: same spatial size; same channels
    def forward(self, x):
        if self.residual_will_be_used:
            return x + self.sequential(x)
        else:
            return self.sequential(x)

# This is the MobileNetV1 block
# Depthwise conv
#       ↓
# Pointwise conv (1x1)
class DepthwiseSeparableConv(nn.Module):
    """Retained from original for compatibility; not used in V3-Small backbone."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise_convolution = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )

        self.pointwise_convolution = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        intermediate = self.depthwise_convolution(x)
        return self.pointwise_convolution(intermediate)


# Implements MobileNetV3-Small architecture
class MobileNet(nn.Module):

    def __init__(self, num_classes=18, width_mult=1.0, dropout_prob=0.2):
        super(MobileNet, self).__init__()

        # ── Initial convolution (stride 2, HardSwish) ───────────────────────
        number_of_output_channels_in_initial_convolution = int(16 * width_mult)

        # This downsamples image:
        # Example: 224x224 → 112x112
        self.initial_convolution = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=number_of_output_channels_in_initial_convolution,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(number_of_output_channels_in_initial_convolution),
            HardSwish(inplace=True)
        )

        # ── MobileNetV3-Small bottleneck configuration ───────────────────────
        # Define the architecture:
        # (in_c, out_c, kernel, stride, expand_ratio, use_se, use_hs)
        configuration = [
            (16,  16,  3, 2, 1,     True,  False),   # bneck1
            (16,  24,  3, 2, 4.5,   False, False),   # bneck2
            (24,  24,  3, 1, 3.67,  False, False),   # bneck3
            (24,  40,  5, 2, 4,     True,  True),    # bneck4
            (40,  40,  5, 1, 6,     True,  True),    # bneck5
            (40,  40,  5, 1, 6,     True,  True),    # bneck6
            (40,  48,  5, 1, 3,     True,  True),    # bneck7
            (48,  48,  5, 1, 3,     True,  True),    # bneck8
            (48,  96,  5, 2, 6,     True,  True),    # bneck9
            (96,  96,  5, 1, 6,     True,  True),    # bneck10
            (96,  96,  5, 1, 6,     True,  True),    # bneck11
        ]

        list_of_layers = []

        # This Loop builds blocks...
        # So network becomes a stack of inverted residual blocks
        for in_c, out_c, kernel, stride, expand_ratio, use_se, use_hs in configuration:
            scaled_number_of_input_channels  = int(in_c  * width_mult)
            scaled_number_of_output_channels = int(out_c * width_mult)
            list_of_layers.append(
                InvertedResidual(
                    in_channels=scaled_number_of_input_channels,
                    out_channels=scaled_number_of_output_channels,
                    kernel_size=kernel,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    use_se=use_se,
                    use_hs=use_hs
                )
            )

        self.sequential = nn.Sequential(*list_of_layers)

        # ── Final convolution (1×1, HardSwish) ──────────────────────────────
        if width_mult > 1.0:
            number_of_output_channels_in_final_convolution = int(576 * width_mult)
        else:
            number_of_output_channels_in_final_convolution = 576

        # Expand feature channels before classification
        self.final_convolution = nn.Sequential(
            nn.Conv2d(
                in_channels=int(96 * width_mult),
                out_channels=number_of_output_channels_in_final_convolution,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(number_of_output_channels_in_final_convolution),
            HardSwish(inplace=True)
        )

        # ── Classifier ───────────────────────────────────────────────────────
        # Convert [B, C, H, W] → [B, C, 1, 1]
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # V3-Small uses an expanded (extra) FC head: 
        #   pool → Conv 1×1 → HS → dropout → FC
        #   Conv2d(576 → 1024)
        self.classifier_head = nn.Sequential(
            nn.Conv2d(
                in_channels=number_of_output_channels_in_final_convolution,
                out_channels=1024,
                kernel_size=1,
                bias=True
            ),
            HardSwish(inplace=True)
        )

        self.dropout = nn.Dropout(dropout_prob)

        self.linear_transformation = nn.Linear(1024, num_classes)

        self.initialize_weights()

    # Conv layers
    # kaiming_normal_: Good for ReLU-like activations.
    # BatchNorm: weight = 1; bias = 0
    # Linear: normal_(0,1)
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 1)
                nn.init.constant_(module.bias, 0)

    # full Forward Pass
    # Output shape: [B, num_classes]
    
    #    x
    #     ↓
    #    initial_convolution
    #     ↓
    #    bottleneck blocks
    #     ↓
    #    final_convolution
    #     ↓
    #    global average pooling
    #     ↓
    #    classifier_head
    #     ↓
    #    flatten
    #     ↓
    #    dropout
    #     ↓
    #    linear layer
    #     ↓
    #    class scores


    def forward(self, x):
        intermediate = self.initial_convolution(x)
        intermediate = self.sequential(intermediate)
        intermediate = self.final_convolution(intermediate)
        intermediate = self.average_pooling(intermediate)
        intermediate = self.classifier_head(intermediate)
        intermediate = torch.flatten(intermediate, start_dim=1)
        intermediate = self.dropout(intermediate)
        return self.linear_transformation(intermediate)
