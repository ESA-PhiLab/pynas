import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split='TrainVal', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and masks.
            split (string): One of ['TrainVal', 'Test'] to specify the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # ==== Define class names and number of classes ====
        self.classes = ['Background', 'BurntArea', 'Cloud', 'Waterbodies']
        self.num_classes = len(self.classes)

        # Getting the list of image and mask paths
        self.image_paths = []
        self.mask_paths = []

        image_dir = os.path.join(self.root_dir, split, 'numpy_images')
        mask_dir = os.path.join(self.root_dir, split, 'numpy_masks')

        image_filenames = os.listdir(image_dir)
        mask_filenames = os.listdir(mask_dir)

        for image_filename in image_filenames:
            if image_filename.endswith('.npy'):
                self.image_paths.append(os.path.join(image_dir, image_filename))

        for mask_filename in mask_filenames:
            if mask_filename.endswith('.npy'):
                self.mask_paths.append(os.path.join(mask_dir, mask_filename))

        # Ensure both image_paths and mask_paths are sorted to match corresponding pairs
        self.image_paths.sort()
        self.mask_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = np.load(img_path)
        if image.shape[-1] == 7:  # Assuming 7 channels
            image = image.transpose(2, 0, 1)
        mask = np.load(mask_path)
        if mask.shape[-1] == 4:  # Assuming 7 channels
            mask = mask.transpose(2, 0, 1)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


class SegmentationDataModule(LightningDataModule):
    def __init__(self, root_dir, batch_size=8, num_workers=1, transform=None, val_split=0.3):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers
        self.val_split = val_split

        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage='fit'):
        # Load the full dataset
        full_dataset = SegmentationDataset(
            root_dir=self.root_dir,
            split='TrainVal',
            transform=self.transform,
        )

        # Store class info for external access
        self.class_names = full_dataset.classes
        self.num_classes = full_dataset.num_classes
        # Calculate shape of input data
        sample, _ = full_dataset[0]
        # print(sample.shape)
        self.input_shape = sample.shape

        # Calculate the number of samples for validation
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size

        # Split the dataset into train and validation
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        # Load test dataset
        if stage == 'test':
            self.test_dataset = SegmentationDataset(
                root_dir=self.root_dir,
                split='Test',
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )