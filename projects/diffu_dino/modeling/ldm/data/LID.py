# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import PIL
import rasterio
import os
import pandas as pd

import albumentations as A
import cv2

class SalObjDataset(Dataset):
	def __init__(self, img_name_list, root ='/scratch/dr/mariam/FINAL_VERSION/crops/', mask_root ='/scratch/dr/mariam/FINAL_VERSION/masks/',transform=None, aug = True, test = False,  version = "train",size = 256):

		self.df = pd.read_csv(img_name_list)
		self.root = root
		self.mask_root = mask_root
		self.paths = self.df['paths']
		# self.paths = self.paths[:50]
		self.transform = transform
		self.aug = aug
		self.test = test
		self.size = size
		self.version = version
														
	def __len__(self):
		return len(self.paths)

	def __getitem__(self,idx):
		src_mask = rasterio.open(os.path.join(self.mask_root + self.paths[idx]))
		im2 = src_mask.read().astype('uint8').transpose((1, 2, 0))
		# mask = im2.repeat(3 , axis = 2)
		mask = im2

		src = rasterio.open(os.path.join(self.root + self.paths[idx]))
		metadata = src.meta.copy()
		im = src.read([1,2,3]).astype(np.float32)
		# im = np.clip(im, 0, 8000)
		imidx = np.array([idx])

		im = im.transpose(1,2,0)
		if self.aug == True:
			transform = A.Compose([
				A.HorizontalFlip(p=0.5),
				A.VerticalFlip(p=0.5),
				A.RandomRotate90(p=0.5),
				A.RandomCrop(256,256),
				# A.ColorJitter(p=0.5),
			])
			transformed = transform(image=im , mask = mask)
			im , mask = transformed['image'] , transformed['mask']
			# mask = mask.transpose(2,0,1)
   
		im = im.transpose(2,0,1)
		im[im > 8000]=1
        #im[im==1]=np.max(im)
        #im= im.astype(np.uint8)
		percentile_values = np.percentile(im, 99, axis=(1, 2))
        # print(percentile_values)
		im[0][(im[0] == 1)] = percentile_values[0]
		im[1][(im[1] == 1)] = percentile_values[1]
		im[2][(im[2] == 1)] = percentile_values[2]
		im[0][(im[0] > percentile_values[0])] = percentile_values[0]
		im[1][(im[1] > percentile_values[1])] = percentile_values[1]
		im[2][(im[2] > percentile_values[2])] = percentile_values[2]
		percentile_values = np.percentile(im, 1, axis=(1, 2))
		im[0][(im[0] < percentile_values[0])] = percentile_values[0]
		im[1][(im[1] < percentile_values[1])] = percentile_values[1]
		im[2][(im[2] < percentile_values[2])] = percentile_values[2]
		
		if np.max(im) > 400:
			im = (im/np.max(im))*255
  
		# maxx[maxx == 0] = 0.02 
		im = im - np.min(im,(1,2)).reshape(3,1,1)
		maxx = np.max(im,(1,2)).reshape(3,1,1)

		im = im / maxx
		if np.isnan(im).any():
			print("error at==================: ", self.paths[idx])
		# mean = [161.4874, 156.3868, 146.1488] 
		# std = [80.1343, 76.3564, 76.4156]     
		im = im.transpose(1, 2 , 0)
		image = (im  * 2 - 1).astype(np.float32)
		# mask = (mask / 127 - 1).astype(np.float32) 
		# image = image.transpose(1,2,0)
		# image = image.astype(np.float32)
		# print(np.min(image) , np.max(image))
		sample = {'imidx':imidx, 'image':image, 'segmentation':mask}
		return sample

class SatTrain(SalObjDataset):
    def __init__(self, **kwargs):
        super().__init__(img_name_list = '/scratch/dr/y.nawar/LID_train.csv', transform=None, aug = True, test = False, version = "train", **kwargs)

class SatValidation(SalObjDataset):
    def __init__(self, **kwargs):
        super().__init__(img_name_list = '/scratch/dr/y.nawar/LID_val.csv', transform=None, aug = True, test = False, version = "validation", **kwargs)



