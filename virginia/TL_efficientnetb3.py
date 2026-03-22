from torchvision import models
import torch.nn as nn


def get_pretrained_model(model_name='efficientnet_b3', num_classes=8, feature_extract=True, dropout_prob=0.3):
    """
    Pretrained EfficientNetB3 for ISIC 2019 classification.
    Mirrors TL_resnet50.py for good comparison.

    Architecture:
        EfficientNetB3 backbone (pretrained on ImageNet, optionally frozen)
        --> Dropout(dropout_prob)
        --> Linear(1536, num_classes)

    Args:
        model_name:      'efficientnet_b3'
        num_classes:     8 for ISIC 2019 ("UNK" removed)
        feature_extract: True  = freeze backbone, train head only
                         False = fine-tune entire network
        dropout_prob:    dropout before final linear layer
    """
    if model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

        # EfficientNetB3 classifier is: [Dropout(0.3), Linear(1536, 1000)]
        # We replace the whole classifier with our own head
        num_in_features = model.classifier[1].in_features  # 1536
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(num_in_features, num_classes)
        )
    else:
        raise ValueError(f"Model {model_name} is unsupported.")

    return model
