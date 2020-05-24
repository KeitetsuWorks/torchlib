##
## @file        custom_voc_detection.py
## @brief       Custom Pascal VOC Detection Dataset Class
## @author      Keitetsu
## @date        2020/05/20
## @copyright   Copyright (c) 2020 Keitetsu
## @par         License
##              This software is released under the MIT License.
##


import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import torchvision
from torchvision.datasets.voc import DATASET_YEAR_DICT, download_extract
from torchvision.datasets.utils import verify_str_arg


class CustomVOCDetection(torchvision.datasets.voc.VOCDetection):

    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 classes=None):
        super(torchvision.datasets.voc.VOCDetection, self).__init__(root, transforms, transform, target_transform)
        self.year = year
        if year == "2007" and image_set == "test":
            year = "2007-test"
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        valid_sets = ["train", "trainval", "val"]
        if year == "2007-test":
            valid_sets.append("test")
        self.image_set = verify_str_arg(image_set, "image_set", valid_sets)
        self.classes = classes
        if self.classes:
            self.custom_image_set_filename = os.path.join(self.root, 'custom_voc_detection_image_set.txt')
        else:
            self.custom_image_set_filename = None

        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Main')

        if self.classes:
            file_count, non_difficult_file_count = self.make_custom_image_set(splits_dir)
            split_f = self.custom_image_set_filename
        else:
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))
        
        if self.classes:
            # オブジェクト統計情報を取得
            obj_count, non_difficult_obj_count = self.get_number_of_objects()
            
            # 統計情報を表示
            print("non-d: non-difficult")
            count_df = pd.DataFrame(
                {
                    'class': self.classes,
                    '# files' : file_count,
                    '# non-d files': non_difficult_file_count,
                    '# objects': obj_count,
                    '# non-d objects': non_difficult_obj_count
                }
            )
            pd.set_option('display.max_rows', None)
            print(count_df)

    def make_custom_image_set(self, splits_dir):
        master_image_set_f = os.path.join(splits_dir, self.image_set.rstrip('\n') + '.txt')
        print("loading file list...: %s" % (master_image_set_f))

        # ファイルリストを作成
        with open(master_image_set_f, mode='rt') as f:
            file_names = [x.strip() for x in f.readlines()]
        print("number of image files: %d" % (len(file_names)))

        # ground_truth_labelを表に集約
        ground_truth_table = []
        for key in self.classes:
            key_image_set_f = os.path.join(splits_dir, key + '_' + self.image_set.rstrip('\n') + '.txt')
            print("loading image set...: %s" % (key_image_set_f))

            with open(key_image_set_f, mode='rt') as f:
                key_image_set_lines = f.readlines()

            ground_truth_labels = []
            for key_image_set_line in key_image_set_lines:
                ground_truth_label = key_image_set_line.split()[-1]
                ground_truth_labels.append(int(ground_truth_label))
            print("number of image files: %d" % (len(ground_truth_labels)))
            ground_truth_table.append(ground_truth_labels)

        # ファイル統計情報を取得
        ground_truth_array = np.array(ground_truth_table)
        # 各クラスのオブジェクトを含むファイル数を取得
        file_count = np.count_nonzero(ground_truth_array != -1, axis = 1)
        # 各クラスオブジェクトを含むファイル数を取得 (non-difficult)
        non_difficult_file_count = np.count_nonzero(ground_truth_array == 1, axis = 1)

        # classesのオブジェクトを含むファイルのみを抽出
        obj_flag_array = np.any(ground_truth_array != -1, axis = 0)

        target_file_names = []
        for i, obj_flag in enumerate(obj_flag_array):
            if (obj_flag == True):
                target_file_names.append(file_names[i])

        with open(self.custom_image_set_filename, mode='wt') as f:
            f.write('\n'.join(target_file_names))
        print("custom image set: %s" % (self.custom_image_set_filename))
        print("number of image files: %d" % (len(target_file_names)))

        return file_count, non_difficult_file_count

    def get_number_of_objects(self):
        obj_count = [0] * len(self.classes)
        non_difficult_obj_count = [0] * len(self.classes)
        for i, annotation_f in enumerate(self.annotations):
            voc_dict = self.parse_voc_xml(ET.parse(annotation_f).getroot())

            for obj in voc_dict["annotation"]["object"]:
                if (obj["name"] in self.classes):
                    classes_index = self.classes.index(obj["name"])
                    
                    obj_count[classes_index] += 1
                    if (int(obj["difficult"]) != 1):
                        non_difficult_obj_count[classes_index] += 1

        return obj_count, non_difficult_obj_count
