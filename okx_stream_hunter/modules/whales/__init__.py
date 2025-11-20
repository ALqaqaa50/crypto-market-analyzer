"""Whale detection module"""

from .detector import WhaleDetector, WhaleEvent
from .orderbook_analyzer import OrderBookWhaleAnalyzer

__all__ = [
    "WhaleDetector",
    "WhaleEvent",
    "OrderBookWhaleAnalyzer",
]