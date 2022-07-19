import random
import cv2
from PIL import Image, ImageFilter
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class MultiCropDataset(datasets.VisionDataset):
    def __init__(
        self,
        dataset,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
        pil_blur=False,
        color_distorsion_scale=1.0,
    ):
        self.dataset = dataset

        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        color_transform = [get_color_distortion(color_distorsion_scale), transforms.RandomApply([RandomGaussianBlur()], p=0.5)]
        if pil_blur:
            color_transform = [get_color_distortion(color_distorsion_scale), transforms.RandomApply([PILRandomGaussianBlur()], p=0.5)]
        trans = []
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.Compose(color_transform),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
                ])
            ] * nmb_crops[i])
        self.trans = trans

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        image, target = self.dataset.__getitem__(index)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class RandomGaussianBlur(object):
    def __call__(self, img):
        # do_it = np.random.rand() > 0.5
        # if not do_it:
        #     return img
        sigma = np.random.rand() * 1.9 + 0.1
        return Image.fromarray(cv2.GaussianBlur(np.asarray(img), (23, 23), sigma))

class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, radius_min=0.1, radius_max=2.):
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
