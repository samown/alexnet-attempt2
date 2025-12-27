"""
AlexNet models for iFood 2019 Challenge
"""

from .alexnet_baseline import AlexNetBaseline
from .alexnet_modified1 import AlexNetModified1
from .alexnet_modified2 import AlexNetModified2
from .alexnet_combined import AlexNetCombined

__all__ = [
    'AlexNetBaseline',
    'AlexNetModified1',
    'AlexNetModified2',
    'AlexNetCombined',
]



