import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import cv2 as cv
from ldm.data.enhancement import imread , preprocess
import albumentations as A

class GlaucomaBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 mode = 'test',
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        
        self.data_paths = txt_file
        self.data_root = data_root
        self.mode = mode
        self.data_frame = pd.read_csv(txt_file)
        self.image_paths = self.data_frame['image_name'].values
        self._length = len(self.image_paths)
        self.label = self.data_frame['label'].values

        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
            'class_label': np.array(self.label),
            'human_label': np.array(["Glaucoma" if l == 1 else "None Glaucoma" for l in self.label])
        }

        self.size = size
        # self.interpolation = {"linear": PIL.Image.LINEAR,
        #                       "bilinear": PIL.Image.BILINEAR,
        #                       "bicubic": PIL.Image.BICUBIC,
        #                       "lanczos": PIL.Image.LANCZOS,
        #                       }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = imread(example["file_path_"])
        image, mask = preprocess(image)
        image = cv.resize(image, (512,512))
        # color_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(color_converted)
        if self.mode == 'train':
            transform = A.ReplayCompose([
				A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5)])
            data = transform(image = image)
            image = data['image']
        im_min = np.min(image)
        im_max = np.max(image)
        image = (image - im_min) / (im_max - im_min)
        example["image"] = (image * 2) - 1
        return example

        
class GlaucomaTrain(GlaucomaBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="/scratch/dr/GlaucomaUnsupervised/Data/CSVs_Train/Train_org/train_full.csv", data_root='/scratch/dr/GlaucomaNewExams/', mode = "train", **kwargs)
        
class GlaucomaValidation(GlaucomaBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="/scratch/dr/GlaucomaUnsupervised/Data/CSVs_Validation/new_val.csv", data_root='/scratch/dr/GlaucomaNewExams/',mode = "valid",
                         flip_p=flip_p, **kwargs)
