import torchvision
from torch import nn

class DocDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # backbone model: MobileNet V3
        self.backbone = torchvision.models.mobilenet_v3_small(pretrained=True)
        self.backbone.classifier = nn.Linear(576, 4*2)

    def forward(self, x):
        return self.backbone(x)