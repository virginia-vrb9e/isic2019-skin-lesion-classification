from torchvision import models
import torch.nn as nn


def get_pretrained_model(model_name='resnet50', num_classes=8, feature_extract=True, dropout_prob=0.3):
    """
    Pretrained ResNet50 for ISIC 2019 classification.

    Architecture:
        ResNet50 backbone (pretrained on ImageNet, optionally frozen)
        --> Dropout(dropout_prob)
        --> Linear(2048, num_classes)

    Args:
        model_name:      'resnet50'
        num_classes:     8 for ISIC 2019
        feature_extract: True  = freeze backbone, train head only
                         False = fine-tune entire network
        dropout_prob:    dropout before final linear layer
    """
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

        num_in_features = model.fc.in_features  # 2048 for ResNet50
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(num_in_features, num_classes)
        )
    else:
        raise ValueError(f"Model {model_name} is unsupported.")

    return model
