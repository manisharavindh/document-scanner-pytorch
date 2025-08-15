import torchvision
from torch import nn

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class DocDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # backbone model: MobileNet V3
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        self.backbone = torchvision.models.mobilenet_v3_small(weights=weights)
        # self.backbone.classifier = nn.Linear(576, 4*2)

        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 4 * 2)  # 8 outputs for 4 corners
        )


    def forward(self, x):
        return self.backbone(x)
    
model = DocDetector()