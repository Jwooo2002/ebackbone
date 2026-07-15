"""Frame-only B0 model definitions.

Production B0 uses an audited local equivalent of the standard torchvision
ResNet-18 topology because torchvision is not a project dependency.  The only
architecture adaptations are the required two-channel input convolution and
the 100-class linear classifier.  Model construction is always random and this
module contains no external-weight or network-loading path.

The small four-convolution model is retained separately for bounded engineering
diagnostics.  It is not a scientific B0 architecture.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


B0_INPUT_SHAPE = (2, 480, 640)
B0_CLASS_COUNT = 100
B0_EMBEDDING_DIM = 512
COMPACT_DEBUG_EMBEDDING_DIM = 64
B0_BOUNDED_CPU_BATCH_SIZE = 1
B0_BOUNDED_CPU_RSS_LIMIT_BYTES = 2_147_483_648
PRODUCTION_MODEL_NAME = "resnet18"
DEBUG_MODEL_NAME = "compact_debug"


class _BasicBlock(nn.Module):
    """Standard ResNet-18 residual basic block."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        *,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs: Tensor) -> Tensor:
        identity = inputs

        output = self.conv1(inputs)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        output += identity
        return self.relu(output)


class ProductionB0ResNet18(nn.Module):
    """Randomly initialized two-channel ResNet-18 for scientific B0."""

    def __init__(self, *, class_count: int = B0_CLASS_COUNT) -> None:
        super().__init__()
        _validate_class_count(class_count)
        if class_count != B0_CLASS_COUNT:
            raise ValueError(f"production B0 class_count must be exactly {B0_CLASS_COUNT}")
        self.class_count = class_count
        self._current_channels = 64
        self.conv1 = nn.Conv2d(
            2,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, block_count=2)
        self.layer2 = self._make_layer(128, block_count=2, stride=2)
        self.layer3 = self._make_layer(256, block_count=2, stride=2)
        self.layer4 = self._make_layer(512, block_count=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(B0_EMBEDDING_DIM, class_count)
        self._initialize_random_weights()

    def _make_layer(
        self,
        channels: int,
        *,
        block_count: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self._current_channels != channels * _BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self._current_channels,
                    channels * _BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(channels * _BasicBlock.expansion),
            )

        blocks = [
            _BasicBlock(
                self._current_channels,
                channels,
                stride=stride,
                downsample=downsample,
            )
        ]
        self._current_channels = channels * _BasicBlock.expansion
        blocks.extend(
            _BasicBlock(self._current_channels, channels)
            for _ in range(1, block_count)
        )
        return nn.Sequential(*blocks)

    def _initialize_random_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def encode(self, event_frame: Tensor) -> Tensor:
        """Return one 512-dimensional embedding for each native event frame."""

        _validate_event_frame(event_frame)
        features = self.conv1(event_frame)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.maxpool(features)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        return self.avgpool(features).flatten(start_dim=1)

    def forward(self, event_frame: Tensor) -> Tensor:
        return self.classifier(self.encode(event_frame))


class CompactDebugB0FrameClassifier(nn.Module):
    """Small frame CNN reserved for engineering and debug checks."""

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
        self.classifier = nn.Linear(COMPACT_DEBUG_EMBEDDING_DIM, class_count)

    def encode(self, event_frame: Tensor) -> Tensor:
        """Return the 64-dimensional diagnostic embedding."""

        _validate_event_frame(event_frame)
        features = self.encoder(event_frame)
        return self.global_average_pool(features).flatten(start_dim=1)

    def forward(self, event_frame: Tensor) -> Tensor:
        return self.classifier(self.encode(event_frame))


_MODEL_BUILDERS: dict[str, Callable[[int], nn.Module]] = {
    PRODUCTION_MODEL_NAME: lambda class_count: ProductionB0ResNet18(class_count=class_count),
    DEBUG_MODEL_NAME: lambda class_count: CompactDebugB0FrameClassifier(class_count=class_count),
}


def build_b0_model(
    model_name: str = PRODUCTION_MODEL_NAME,
    *,
    class_count: int = B0_CLASS_COUNT,
) -> nn.Module:
    """Build an explicitly named, randomly initialized B0 model."""

    try:
        builder = _MODEL_BUILDERS[model_name]
    except KeyError as exc:
        raise ValueError(f"unsupported B0 model: {model_name}") from exc
    return builder(class_count)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def native_input_batch_bytes(batch_size: int) -> int:
    """Dense float32 bytes for one native-resolution B0 input batch."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    channels, height, width = B0_INPUT_SHAPE
    return batch_size * channels * height * width * torch.float32.itemsize


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
    "COMPACT_DEBUG_EMBEDDING_DIM",
    "CompactDebugB0FrameClassifier",
    "DEBUG_MODEL_NAME",
    "PRODUCTION_MODEL_NAME",
    "ProductionB0ResNet18",
    "build_b0_model",
    "native_input_batch_bytes",
    "trainable_parameter_count",
]
