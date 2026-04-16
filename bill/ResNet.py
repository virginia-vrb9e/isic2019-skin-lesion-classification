import torch.nn as nn
import torch

'''
ResNet code converted to ResNet-50, which uses Bottleneck blocks (1x1 → 3x3 → 1x1 convolutions) 
  instead of BasicBlocks, with 3-4-6-3 layer configuration:
Key differences from your BasicBlock ResNet:
	            Your ResNet             ResNet-50
Block type	    BasicBlock (2 convs)	Bottleneck (3 convs)
Conv pattern	3×3 → 3×3	            1×1 → 3×3 → 1×1
Expansion	    ×1	                    ×4 (channels expand at end)
Layer config	[2, 2, 2, 2] blocks	    [3, 4, 6, 3] blocks
Final channels	512	                    2048 (512 × 4)

The expansion = 4 is the core idea: each Bottleneck compresses channels with the first 1×1 conv, 
does spatial work with the 3×3, 
then expands back out with the final 1×1 — making deeper networks more efficient.
'''

'''
Big Picture Summary
    • Input image → stem (conv + pool)
    • Pass through 4 residual stages
    • Each stage uses Bottleneck blocks
    • Channels grow:
        64 → 256 → 512 → 1024 → 2048
    • Global average pooling
    • Fully connected layer → predictions
'''

import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4  # Output channels are 4x the intermediate channels

    # •	in_channels: input depth
    # •	out_channels: base channels
    # •	stride: controls downsampling
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()

        # 1x1 conv to reduce channels, no bias
        self.convolution_1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, bias=False
        )

        # normalize for stability
        self.batch_normalization_1 = nn.BatchNorm2d(out_channels)

        # 3x3 conv (spatial processing)
        # •	stride may downsample (reduce width/height)
        # •	padding=1 keeps size consistent (if stride=1)
        self.convolution_2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.batch_normalization_2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv to expand channels
        # expands cnannels by 4x
        self.convolution_3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, stride=1, bias=False
        )
        self.batch_normalization_3 = nn.BatchNorm2d(out_channels * self.expansion)

        # ReLu activation
        self.relu = nn.ReLU(inplace=True)

        # Shortcut: project identity if shape changes
        self.shortcut = nn.Sequential()
        # If shape mismatch:
        #   spatial size changed (stride ≠ 1)
        #   channel count changed
        # self.shortcut = nn.Sequential(... • Projects input to match output shape
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * self.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    # Forward Pass (Bottleneck): Defines how data flows
    def forward(self, x):
        intermediate = self.convolution_1(x)
        intermediate = self.batch_normalization_1(intermediate)
        intermediate = self.relu(intermediate)

        intermediate = self.convolution_2(intermediate)
        intermediate = self.batch_normalization_2(intermediate)
        intermediate = self.relu(intermediate)

        intermediate = self.convolution_3(intermediate)
        intermediate = self.batch_normalization_3(intermediate)

        # residual connection: Adds input (skip connection)
        intermediate = intermediate + self.shortcut(x)
        # final activation
        return self.relu(intermediate)


class ResNet50(nn.Module):

    # Output classes = 18
    def __init__(self, num_classes=18):
        super(ResNet50, self).__init__()

        # Stem: 7x7 conv, BN, ReLU
        # Stem layer
        # self.convolution...initial feature extractor
        # nn.Conv2d(...Input: RGB image (3 channels)
        #              Output: 64 channels
        #              Large kernel (7×7)
        #              Downsamples
 
        self.convolution = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # max pool: further downsampling
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-50 layer config: [3, 4, 6, 3] blocks
        # in_channels for each layer accounts for Bottleneck expansion (x4)
        # No downsampling, output channels = 256 (64 × 4)
        self.residual_layer_1 = self._make_layer(64,  64,  num_blocks=3, stride=1)
        # Downsampling; Output = 512
        self.residual_layer_2 = self._make_layer(256, 128, num_blocks=4, stride=2)
        # Output = 1024
        self.residual_layer_3 = self._make_layer(512, 256, num_blocks=6, stride=2)
        # Output = 2048
        self.residual_layer_4 = self._make_layer(1024, 512, num_blocks=3, stride=2)

        # Converts feature map → 1×1; works for any input size
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # Final expanded channels = 512 * 4 = 2048
        # Output: class scores
        self.linear_transformation = nn.Linear(512 * Bottleneck.expansion, num_classes)

    # Builds a stack of Bottleneck blocks
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        list_of_blocks = []

        # First block handles stride and channel projection
        list_of_blocks.append(Bottleneck(in_channels, out_channels, stride))

        # Remaining blocks: input is now out_channels * expansion
        for _ in range(1, num_blocks):
            list_of_blocks.append(
                Bottleneck(out_channels * Bottleneck.expansion, out_channels)
            )

        # Converts list → sequential module
        return nn.Sequential(*list_of_blocks)

    def forward(self, x):

        # Stem
        intermediate = self.convolution(x)
        intermediate = self.max_pooling(intermediate)

        # Residual layers
        intermediate = self.residual_layer_1(intermediate)
        intermediate = self.residual_layer_2(intermediate)
        intermediate = self.residual_layer_3(intermediate)
        intermediate = self.residual_layer_4(intermediate)

        # Converts [batch, channels, 1, 1] → [batch, channels]
        intermediate = self.average_pooling(intermediate)
        intermediate = torch.flatten(intermediate, start_dim=1)
        # Produces class scores
        return self.linear_transformation(intermediate)
