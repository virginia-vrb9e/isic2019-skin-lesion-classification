from torchvision import models
import torch.nn as nn


def get_pretrained_model(model_name='mobilenet_v2', num_classes=8, feature_extract=True, dropout_prob=0.3):
    """
    Pretrained MobileNetV2 for ISIC 2019 classification.
    Mirrors TL_resnet50.py exactly.

    Architecture:
        MobileNetV2 backbone (pretrained on ImageNet, optionally frozen)
        --> Dropout(dropout_prob)
        --> Linear(1280, num_classes)

    Args:
        model_name:      'mobilenet_v2'
        num_classes:     8 for ISIC 2019 (sans "UNK")
        feature_extract: True  = freeze backbone, train head only
                         False = fine-tune entire network
        dropout_prob:    dropout before final linear layer
    """
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

        # MobileNetV2 classifier is: [Dropout(0.2), Linear(1280, 1000)]
        # We replace the whole classifier with our own head
        num_in_features = model.classifier[1].in_features  # 1280
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(num_in_features, num_classes)
        )
    else:
        raise ValueError(f"Model {model_name} is unsupported.")

    return model
