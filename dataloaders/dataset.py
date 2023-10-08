import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler


class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
        sign=1,
    ):
        self._base_dir = base_dir
        self.digit_str = self._base_dir.split('/')[-1].split('.')[0][-1]
        self._base_dir_data = base_dir.split('/cross')[0]
        self.sample_list = []
        self.unlabel= []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir, "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            if sign==1:
                new_filename = self._base_dir.replace("train", "untrain")
                with open(new_filename, "r") as f2:
                    self.unlabel = f2.readlines()
                self.unlabel = [item.replace("\n", "") for item in self.unlabel]
                self.sample_list.extend(self.unlabel)

        elif self.split == "val":
            with open(self._base_dir, "r") as f:
            # with open(self._base_dir + "/test1.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        # print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            if case.split('_')[0]!='new':
                h5f = h5py.File(self._base_dir_data + f"/data/FL/FL_slices3/{case}.h5", "r")
            else:
                h5f = h5py.File(self._base_dir_data + f"/data/FL/FL_unlabel_silces3/{case}.h5", "r")
        else:
            h5f = h5py.File(self._base_dir_data + f"/data/FL/FL_slices3/{case}.h5", "r")
        image = h5f["image"][:]
        label1 = h5f["label1"][:]
        label2 = h5f["label2"][:]
        if self.digit_str == '2':
            image = cv2.copyMakeBorder(image, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=0)
            label1 = cv2.copyMakeBorder(label1, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=0)
            label2 = cv2.copyMakeBorder(label2, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=0)

        elif self.digit_str == '3':
            image = cv2.copyMakeBorder(image, 12, 20, 12, 20, cv2.BORDER_CONSTANT, value=0)
            label1 = cv2.copyMakeBorder(label1, 12, 20, 12, 20, cv2.BORDER_CONSTANT, value=0)
            label2 = cv2.copyMakeBorder(label2, 12, 20, 12, 20, cv2.BORDER_CONSTANT, value=0)

        sample = {"image": image, "label1": label1,"label2": label2}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx

        return sample


def random_rot_flip(image, label1=None,label2=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label1 is not None and label2 is not None:
        label1 = np.rot90(label1, k)
        label1 = np.flip(label1, axis=axis).copy()
        label2 = np.rot90(label2, k)
        label2 = np.flip(label2, axis=axis).copy()
        return image, label1, label2
    else:
        return image


def random_rotate(image, label1, label2):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label1 = ndimage.rotate(label1, angle, order=0, reshape=False)
    label2 = ndimage.rotate(label2, angle, order=0, reshape=False)
    return image, label1, label2


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label1, label2 = sample["image"], sample["label1"], sample["label2"]

        if random.random() > 0.5:
            image, label1, label2 = random_rot_flip(image, label1, label2)
        elif random.random() <= 0.5:
            image, label1, label2 = random_rotate(image, label1, label2)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label1 = zoom(label1, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label2 = zoom(label2, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label1 = torch.from_numpy(label1.astype(np.uint8))
        label2 = torch.from_numpy(label2.astype(np.uint8))
        sample = {"image": image, "label1": label1, "label2": label2}
        return sample

class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label1, label2 = sample["image"], sample["label1"], sample["label2"]
        image = self.resize(image)
        label1 = self.resize(label1)
        label2 = self.resize(label2)
        # weak augmentation is rotation / flip
        image_weak, label1, label2 = random_rot_flip(image, label1, label2)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label1 = torch.from_numpy(label1.astype(np.uint8))
        label2 = torch.from_numpy(label2.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label1_aug": label1,
            "label2_aug": label2,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        #label,unlabel,18,12每个epoch给多少个label
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0


    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        #ABC->A,B,C
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

def get_dataset(name,sign,patch_size):
    if name == 'huaxiFG':
        dir_img1 = r'../data/huaxi_img/cross_FL/train1.list'
        dir_v1 = r'../data/huaxi_img/cross_FL/test1.list'

        dir_img2 = r'../data/huaxi_img/cross_FL/train2.list'
        dir_v2 = r'../data/huaxi_img/cross_FL/test2.list'

        dir_img3 = r'../data/huaxi_img/cross_FL/train3.list'
        dir_v3 = r'../data/huaxi_img/cross_FL/test3.list'

        train_dataset_1 = BaseDataSets(base_dir=dir_img1, split="train", num=None, sign=sign,
                                       transform=transforms.Compose([RandomGenerator(patch_size)]))

        train_dataset_2 = BaseDataSets(base_dir=dir_img2, split="train", num=None, sign=sign,
                                       transform=transforms.Compose([RandomGenerator(patch_size)]))

        train_dataset_3 = BaseDataSets(base_dir=dir_img3, split="train", num=None, sign=sign,
                                       transform=transforms.Compose([RandomGenerator(patch_size)]))

        eval_dataset1 = BaseDataSets(base_dir=dir_v1, split="val")

        eval_dataset2 = BaseDataSets(base_dir=dir_v2, split="val")

        eval_dataset3 = BaseDataSets(base_dir=dir_v3, split="val")

        return [train_dataset_1, train_dataset_2, train_dataset_3], [eval_dataset1, eval_dataset2, eval_dataset3]
    elif name == 'huaxiFGOnly':

        dir_img1 = r'../data/huaxi_img/cross/train_slices.list'
        dir_v1 = r'../data/huaxi_img/cross/test.list'

        train_dataset = BaseDataSets(base_dir=dir_img1, split="train", num=None, sign=sign,
                                       transform=transforms.Compose([RandomGenerator(patch_size)]))

        eval_dataset = BaseDataSets(base_dir=dir_v1, split="val")
        print(len(train_dataset),len(eval_dataset))
        return [train_dataset], [eval_dataset]

if __name__ == '__main__':
    if __name__ == '__main__':
        a, b = get_dataset('huaxiFGOnly', 0, 140, [256, 256])
        print(len(a))
        print(len(b))
    # labeled_idxs = list(range(300))
    # unlabeled_idxs = list(range(300,740))
    # batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, 18, 12)
    # for _ in range(3):
    #     i = 0
    #     for x in batch_sampler:
    #         i += 1
    #         print('%02d' % i, '\t', x)