# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import os
import torch.utils.data as data
from PIL import Image


"""# Load Dataset"""

class Dataset(data.Dataset):

    def __init__(self, root='./', load_set='train', transform=None):
        self.root = root#os.path.expanduser(root)
        self.transform = transform
        self.load_set = load_set  # 'train','val','test'
        print(root)
        self.images = np.load(os.path.join(root, 'images-%s.npy'%self.load_set))
        self.depths = np.load(os.path.join(root, 'images-d-%s.npy'%self.load_set))
        self.points2d = np.load(os.path.join(root, 'points2d-%s.npy'%self.load_set))
        self.points3d = np.load(os.path.join(root, 'points3d-%s.npy'%self.load_set))
        
        #if shuffle:
        #    random.shuffle(data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, points2D, points3D).
        """
        rgb_path = self.images[index].replace('/home/razvan/HUP-3D-model/data/', '/home/users/caramr2/hup3d/hup3d-data/')
        image = Image.open(rgb_path)
        depth_path = self.depths[index].replace('/home/razvan/HUP-3D-model/data/', '/home/users/caramr2/hup3d/hup3d-data/')
        depth_path = depth_path.replace('depth_', 'img_depth_')
        depth = Image.open(depth_path)
        point2d = self.points2d[index]
        point3d = self.points3d[index]
        # print(self.images[index])
        if self.transform is not None:
            image = self.transform(image)
            depth = self.transform(depth)

        return image[:3], depth,  point2d, point3d

    def __len__(self):
        return len(self.images)
