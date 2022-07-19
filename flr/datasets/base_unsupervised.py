
import numpy as np
import io
import h5py
from PIL import Image
from torch.utils.data import Dataset
from os.path import isfile, isdir, split

class DatasetBaseUnsupervised_HDF5(Dataset):
    def __init__(self, root, transform=None):
        assert isfile(root), '{} does not exist!'.format(root)
        self.root = root

        with h5py.File(self.root, 'r') as f:
            length = len(f['images'])
            # get the folder names
        #     folders = sorted(np.unique(np.asarray(f['unique_foldernames']))) #unique_
        #     class_to_idx = {cls_name: i for i, cls_name in enumerate(folders)}
        # self.class_to_idx = class_to_idx
        self.length = length
        self.transform = transform

        self.h5py_file = None

        print('Base unsupervised dataloader intialised succesfully')

    def __getitem__(self, index):
        if self.h5py_file is None:
            self.h5py_file = h5py.File(self.root, 'r')

        image = Image.open(io.BytesIO(np.array(self.h5py_file['images'][index])))
        # bboxes
        #foldername = f['foldernames'][index]

        # repeat channel for grayscale image
        if image.mode != 'RGB':
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = 0 #self.class_to_idx[foldername]
        return image, label

    def __len__(self):
        return self.length