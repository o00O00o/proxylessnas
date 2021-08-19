import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data_providers.base_provider import *


class Cifar10DataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=64, test_batch_size=64, valid_size=0.5, n_worker=6):

        self._save_path = save_path
        train_dataset = datasets.CIFAR10(self.save_path, transform=transforms.Compose([
            transforms.RandomCrop(self.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ]))

        if valid_size is not None:
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(train_dataset))
            else:
                assert isinstance(valid_size, int), 'invalid valid_size: %s' % valid_size

            num_train = len(train_dataset)
            indices = list(range(num_train))
            split = int(np.floor(valid_size * num_train))

            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train])

            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                train_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = self.train
        else:
            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = None

    @staticmethod
    def name():
        return 'cifar10'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/home/gaoyibo/Datasets/cifar-10/'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download cifar10')

    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])

    @property
    def resize_value(self):
        return 32

    @property
    def image_size(self):
        return 32

