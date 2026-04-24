import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from dataloader import SegmentationDataset


def _make_split(root):
    image_dir = root / "TrainVal" / "numpy_images"
    mask_dir = root / "TrainVal" / "numpy_masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    return image_dir, mask_dir


def test_segmentation_dataset_pairs_files_by_shared_stem(tmp_path):
    image_dir, mask_dir = _make_split(tmp_path)
    np.save(image_dir / "sample.npy", np.zeros((2, 3, 7), dtype=np.float32))
    np.save(mask_dir / "sample.npy", np.zeros((2, 3, 4), dtype=np.float32))

    dataset = SegmentationDataset(tmp_path, split="TrainVal")
    image, mask = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (7, 2, 3)
    assert mask.shape == (4, 2, 3)


def test_segmentation_dataset_rejects_mismatched_image_mask_pairs(tmp_path):
    image_dir, mask_dir = _make_split(tmp_path)
    np.save(image_dir / "image_only.npy", np.zeros((2, 3, 7), dtype=np.float32))
    np.save(mask_dir / "mask_only.npy", np.zeros((2, 3, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="do not match"):
        SegmentationDataset(tmp_path, split="TrainVal")
