import torch.nn as nn
from src.config import IMAGE_HEIGHT, IMAGE_WIDTH, SEQ_LENGTH, NUM_CLASSES

class CaptchaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMAGE_HEIGHT // 4) * (IMAGE_WIDTH // 4), 512), nn.ReLU(),
            nn.Linear(512, SEQ_LENGTH * NUM_CLASSES)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x.view(-1, SEQ_LENGTH, NUM_CLASSES)
