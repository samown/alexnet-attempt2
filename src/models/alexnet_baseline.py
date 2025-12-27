"""
Baseline AlexNet implementation for iFood 2019 Challenge
Original AlexNet architecture with ReLU activation and dropout
"""

import torch
import torch.nn as nn


class AlexNetBaseline(nn.Module):
    """
    Baseline AlexNet architecture
    
    Architecture:
    - 5 Convolutional layers
    - 3 Fully Connected layers
    - ReLU activation
    - Dropout (0.5) in FC layers
    - Output: 251 classes (iFood 2019)
    """
    
    def __init__(self, num_classes: int = 251, dropout: float = 0.5):
        """
        Initialize AlexNet baseline model
        
        Args:
            num_classes: Number of output classes (default: 251 for iFood 2019)
            dropout: Dropout probability for FC layers (default: 0.5)
        """
        super(AlexNetBaseline, self).__init__()
        
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Feature extraction (Convolutional layers)
        self.features = nn.Sequential(
            # Conv1: 224x224x3 -> 55x55x96
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 55x55x96 -> 27x27x96
            
            # Conv2: 27x27x96 -> 27x27x256
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27x256 -> 13x13x256
            
            # Conv3: 13x13x256 -> 13x13x384
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 13x13x384 -> 13x13x384
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 13x13x384 -> 13x13x256
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13x13x256 -> 6x6x256
        )
        
        # Classifier (Fully Connected layers)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten: (batch_size, 256, 6, 6) -> (batch_size, 9216)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
