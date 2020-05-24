##
## @file        ImageDirectoryDataset.py
## @brief       Image Directory Dataset Class
## @author      Keitetsu
## @date        2020/05/20
## @copyright   Copyright (c) 2020 Keitetsu
## @par         License
##              This software is released under the MIT License.
##


import torch

import os
import glob

from PIL import Image


class ImageDirectoryDataset(torch.utils.data.Dataset):

    def __init__(self, list_path, transform=None):
        self.transform = transform
        self._load_list(list_path)
        self.data_num = len(self.data)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        image = Image.open(self.data[idx])
        if (image.mode != 'RGB'):
            image = image.convert('RGB')
        out_data = image
        out_label =  self.label[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label

    def _load_list(self, list_path):
        print("loading list of directories...")
        with open(list_path, mode='rt') as f:
            list_lines = f.readlines()

        self.data = []
        self.label = []
        self.classes = []
        for list_line in list_lines:
            list_line_elements = list_line.split()
            if (len(list_line_elements) == 2):
                dir_path, key = list_line_elements
            else:
                continue

            file_search_path = os.path.join(dir_path, '*')
            file_paths = sorted(glob.glob(file_search_path))

            if (not(key in self.classes)):
                self.classes.append(key)
            classes_index = self.classes.index(key)
            print("class: %s, id: %d, directory: %s, number of files: %d"
                  % (key, classes_index, dir_path, len(file_paths)))

            self.data += file_paths
            self.label += [classes_index] * len(file_paths)

        return
