import torch

from pynas.train.custom_iou import calculate_iou
from pynas.train.losses import CategoricalCrossEntropyLoss, FocalLoss
from pynas.train.mean_squared_error import MeanSquaredError


def test_categorical_cross_entropy_accepts_one_hot_segmentation_targets():
    logits = torch.tensor(
        [
            [
                [[2.0, 0.5], [0.2, 1.0]],
                [[0.1, 1.5], [1.8, 0.3]],
            ]
        ]
    )
    targets = torch.nn.functional.one_hot(torch.tensor([[[0, 1], [1, 0]]]), num_classes=2).permute(
        0, 3, 1, 2
    )

    loss = CategoricalCrossEntropyLoss()(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_focal_loss_returns_scalar_for_one_hot_targets():
    logits = torch.randn(2, 3, 4, 4)
    class_targets = torch.randint(0, 3, (2, 4, 4))
    targets = torch.nn.functional.one_hot(class_targets, num_classes=3).permute(0, 3, 1, 2)

    loss = FocalLoss()(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_calculate_iou_returns_per_sample_values():
    class_targets = torch.tensor([[[0, 1], [1, 0]]])
    targets = torch.nn.functional.one_hot(class_targets, num_classes=2).permute(0, 3, 1, 2)
    logits = targets.float() * 10.0

    iou = calculate_iou(logits, targets, num_classes=2)

    assert iou.shape == (1,)
    assert torch.allclose(iou, torch.ones_like(iou))


def test_calculate_iou_infers_num_classes_from_logits():
    class_targets = torch.ones((1, 2, 2), dtype=torch.long)
    targets = torch.nn.functional.one_hot(class_targets, num_classes=2).permute(0, 3, 1, 2)
    logits = torch.zeros((1, 2, 2, 2))
    logits[:, 0] = 10.0

    iou = calculate_iou(logits, targets)

    assert iou.item() < 1e-4


def test_mean_squared_error_metric_accumulates_state():
    metric = MeanSquaredError()
    metric.update(torch.tensor([1.0, 3.0]), torch.tensor([1.0, 1.0]))

    assert torch.allclose(metric.compute(), torch.tensor(2.0))
