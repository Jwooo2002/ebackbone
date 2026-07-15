"""Minimal production B0 frame-only model.

The module has one constructor and no pretrained-weight loading path.  The
encoder consumes the fixed D012 native-resolution event frame, reduces spatial
cost with explicit strided convolutions, emits one sample embedding through
global average pooling, and applies exactly one linear classifier.
"""

from __future__ import annotations

from torch import Tensor, nn


B0_INPUT_SHAPE = (2, 480, 640)
B0_CLASS_COUNT = 100
B0_EMBEDDING_DIM = 64
B0_BOUNDED_CPU_BATCH_SIZE = 4
B0_BOUNDED_CPU_RSS_LIMIT_BYTES = 1_073_741_824
PRODUCTION_MODEL_NAME = "compact_frame_cnn_v1"


class B0FrameClassifier(nn.Module):
    """Random-initialized compact CNN for one native-resolution event frame."""

    def __init__(self, *, class_count: int = B0_CLASS_COUNT) -> None:
        super().__init__()
        _validate_class_count(class_count)
        self.class_count = class_count
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=7, stride=4, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(B0_EMBEDDING_DIM, class_count)

    def encode(self, event_frame: Tensor) -> Tensor:
        """Return one 64-dimensional embedding for each input sample."""

        _validate_event_frame(event_frame)
        features = self.encoder(event_frame)
        return self.global_average_pool(features).flatten(start_dim=1)

    def forward(self, event_frame: Tensor) -> Tensor:
        return self.classifier(self.encode(event_frame))


def build_b0_model(
    model_name: str = PRODUCTION_MODEL_NAME,
    *,
    class_count: int = B0_CLASS_COUNT,
) -> B0FrameClassifier:
    if model_name != PRODUCTION_MODEL_NAME:
        raise ValueError(f"unsupported B0 model: {model_name}")
    return B0FrameClassifier(class_count=class_count)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def native_input_batch_bytes(batch_size: int) -> int:
    """Dense float32 bytes for one native-resolution B0 input batch."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    channels, height, width = B0_INPUT_SHAPE
    return batch_size * channels * height * width * 4


def _validate_event_frame(event_frame: Tensor) -> None:
    if event_frame.ndim != 4 or tuple(event_frame.shape[1:]) != B0_INPUT_SHAPE:
        raise ValueError(
            "B0 event_frame must have shape [B, 2, 480, 640]; "
            f"got {tuple(event_frame.shape)}"
        )


def _validate_class_count(class_count: int) -> None:
    if isinstance(class_count, bool) or not isinstance(class_count, int) or class_count <= 1:
        raise ValueError("class_count must be an integer greater than one")


__all__ = [
    "B0_BOUNDED_CPU_BATCH_SIZE",
    "B0_BOUNDED_CPU_RSS_LIMIT_BYTES",
    "B0_CLASS_COUNT",
    "B0_EMBEDDING_DIM",
    "B0_INPUT_SHAPE",
    "B0FrameClassifier",
    "PRODUCTION_MODEL_NAME",
    "build_b0_model",
    "native_input_batch_bytes",
    "trainable_parameter_count",
]
