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

#==========================dataset load==========================
class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx, 'image':img,'label':lbl}

class Rescale(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		if random.random() >= 0.5:
			image = image[::-1].copy()
			label = label[::-1].copy()

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		img = transform.resize(image,(new_h,new_w),mode='constant')
		lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx, 'image':img,'label':lbl}

class RandomCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		if random.random() >= 0.5:
			image = image[::-1].copy()
			label = label[::-1].copy()

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h, left: left + new_w]
		label = label[top: top + new_h, left: left + new_w]

		return {'imidx':imidx,'image':image, 'label':label}

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):

		imidx, image, label = sample['imidx'], sample['image'], sample['label']
		tmpImg = np.zeros((image.shape[0],image.shape[1],3))
		tmpLbl = np.zeros(label.shape)

		image = image/np.max(image)
		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		if image.shape[2]==1:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
		else:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
			tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}

class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		imidx, image, label =sample['imidx'], sample['image'], sample['label']

		tmpLbl = np.zeros(label.shape)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		# change the color space
		if self.flag == 2: # with rgb and Lab colors
			tmpImg = np.zeros((image.shape[0],image.shape[1],6))
			tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
			if image.shape[2]==1:
				tmpImgt[:,:,0] = image[:,:,0]
				tmpImgt[:,:,1] = image[:,:,0]
				tmpImgt[:,:,2] = image[:,:,0]
			else:
				tmpImgt = image
			tmpImgtl = color.rgb2lab(tmpImgt)

			# nomalize image to range [0,1]
			tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
			tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
			tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
			tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
			tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
			tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

		elif self.flag == 1: #with Lab color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))

			if image.shape[2]==1:
				tmpImg[:,:,0] = image[:,:,0]
				tmpImg[:,:,1] = image[:,:,0]
				tmpImg[:,:,2] = image[:,:,0]
			else:
				tmpImg = image

			tmpImg = color.rgb2lab(tmpImg)

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
		
		elif self.flag == 66:
			# print("h "*10)
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))
			image = image
			if image.shape[2]==1:
				tmpImg[:,:,0] = (image[:,:,0])
				tmpImg[:,:,1] = (image[:,:,0])
				tmpImg[:,:,2] = (image[:,:,0])
			else:
				tmpImg[:,:,0] = (image[:,:,0])
				tmpImg[:,:,1] = (image[:,:,1])
				tmpImg[:,:,2] = (image[:,:,2])
			
		elif self.flag == 10:
			# print("h "*10)
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))
			image = (image-[161.4874, 156.3868, 146.1488])/[57.3920, 56.4577, 54.1426]
			if image.shape[2]==1:
				tmpImg[:,:,0] = (image[:,:,0])
				tmpImg[:,:,1] = (image[:,:,0])
				tmpImg[:,:,2] = (image[:,:,0])
			else:
				tmpImg[:,:,0] = (image[:,:,0])
				tmpImg[:,:,1] = (image[:,:,1])
				tmpImg[:,:,2] = (image[:,:,2])
		elif self.flag == 50:
			# print("h "*10)
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))
			image = image/255
			if image.shape[2]==1:
				tmpImg[:,:,0] = (image[:,:,0])
				tmpImg[:,:,1] = (image[:,:,0])
				tmpImg[:,:,2] = (image[:,:,0])
			else:
				tmpImg[:,:,0] = (image[:,:,0])
				tmpImg[:,:,1] = (image[:,:,1])
				tmpImg[:,:,2] = (image[:,:,2])

		else: # with rgb color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))
			print("not here "*10)
			##
			if np.max(image)==0:
				print("error "*10)
			##
			else:
				image = image/np.max(image)
			if image.shape[2]==1:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
			else:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
				tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}

class SalObjDataset(Dataset):
	def __init__(self, img_name_list, root ='/scratch/dr/y.nawar/reqaba_all/',transform=None, aug = True, test = False,  version = "train",size = 256):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.df = pd.read_csv(img_name_list)
		# if version == 'train':
		# 	self.df = self.df[:31000]
		self.root = root
		self.image_name_list = [os.path.join(self.root , path) for path in self.df['path'].values]
		self.transform = transform
		self.aug = aug
		self.test = test
		self.size = size
		self.version = version
		interpolation_fn = {
        "cv_nearest": cv2.INTER_NEAREST,
        "cv_bilinear": cv2.INTER_LINEAR,
        "cv_bicubic": cv2.INTER_CUBIC,
        "cv_area": cv2.INTER_AREA,
        "cv_lanczos": cv2.INTER_LANCZOS4,
        "pil_nearest": PIL.Image.NEAREST,
        "pil_bilinear": PIL.Image.BILINEAR,
        "pil_bicubic": PIL.Image.BICUBIC,
        "pil_box": PIL.Image.BOX,
        "pil_hamming": PIL.Image.HAMMING,
        "pil_lanczos": PIL.Image.LANCZOS,
        }['pil_lanczos']
		self.degradation_process = A.SmallestMaxSize(max_size = 110,
                                                        interpolation=interpolation_fn)
														
	def __len__(self):
		return len(self.df)

	def __getitem__(self,idx):
		src = rasterio.open(self.image_name_list[idx])

		# New Inference normalization
		metadata = src.meta.copy()
		im = src.read([1,2,3])  #.astype(np.int32)
		imidx = np.array([idx])
		# print(np.min(im),np.max(im))
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
		# print(np.min(im,(1,2)) , np.max(im,(1,2)))
		if np.max(im)>400:
			print("this should be low resolution data ****************************************")
			print(self.image_name_list[idx])
			im = (im/np.max(im))*255

		im = im - np.min(im,(1,2)).reshape(3,1,1)
		im = im / np.max(im,(1,2)).reshape(3,1,1)
		image = im.transpose(1, 2, 0)

		if self.aug == True:
			transform = A.Compose([
				A.HorizontalFlip(p=0.5),
				A.VerticalFlip(p=0.5),
				A.RandomRotate90(p=0.5),
				# A.RandomCrop(256,256),
				# A.ColorJitter(p=0.5),
			])
			transformed = transform(image=image)
			image = transformed['image']
        
		image = (image * 2) - 1
		# mean = [161.4874, 156.3868, 146.1488] 
		# std = [80.1343, 76.3564, 76.4156]     
		# image = np.moveaxis(((image - mean) / std).astype(np.float32) , -1 , 0) 
		
		image = image.astype(np.float32)
		# print(np.min(image) , np.max(image))
		sample = {'imidx':imidx, 'image':image, 'image_path' : self.image_name_list[idx]}
		return sample

class SalTrain(SalObjDataset):
    def __init__(self, **kwargs):
        super().__init__(img_name_list = '/scratch/dr/y.nawar/AE_train_new_gedan.csv', transform=None, aug = True, test = False, version = "train", **kwargs)

class SalValidation(SalObjDataset):
    def __init__(self, **kwargs):
        super().__init__(img_name_list = '/scratch/dr/y.nawar/AE_valid_new.csv', transform=None, aug = False, test = False, version = "validation", **kwargs)



