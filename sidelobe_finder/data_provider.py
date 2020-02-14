#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import os
import sys
import subprocess
import string
import time
import signal
from threading import Thread
import datetime
import numpy as np
import random
import math
import logging
import copy
import itertools
import pprint


## ADDON ML MODULES
from keras import backend as K
import cv2

## PACKAGE MODULES
from .utils import Utils

##############################
##     GLOBAL VARS
##############################
logger = logging.getLogger(__name__)


##############################
##     CLASS DEFINITIONS
##############################
class DataProvider(object):
	""" Class to read train data from disk and provide to network """

	def __init__(self,datalist,config):
		""" Return a DataProvider object """

		# - Data options
		self.C= config
		self.datalist= datalist	
		self.img_height= 0
		self.img_width= 0
		self.img_depth= 0
		
		# - Options
		self.is_gray_level_img= False
		self.img_channel_mean= [0,0,0]
	
		# - Image data
		self.found_bg = False
		self.all_data= []
		self.classes_count = {}
		self.classes= {}
		self.class_cycle= None
		self.curr_class= None
		self.class_mapping = {}
		self.rgb_img_formats = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'}


	#================================
	##     READ IMAGE
	#================================
	def read_img(self,filename):
		""" Read image data """

		# - Read image according to file extension detected
		file_ext= Utils.get_filename_ext(filename)

		if file_ext=='.fits':
			res= Utils.read_fits(
				filename,
				stretch=self.C.apply_zscale,
				normalize=self.C.normalize_data,
				convertToRGB=self.C.convert_to_rgb
			)
			#if not all(res):
			if not res:
				logger.error("Failed to read FITS file %s!" % (filename))
				return None

			img= res[0]

		elif file_ext in self.rgb_img_formats:
			img = cv2.imread(filename)
		else:
			logger.error("Invalid/unsupported image format given (%s)!" % (file_ext))
			return None

		return img

	#================================
	##     READ DATA LIST
	#================================
	def read_data(self):
		""" Read data list """

		all_imgs = {}
		r_chan_all= np.zeros(0)
		g_chan_all= np.zeros(0)
		b_chan_all= np.zeros(0)	

		# - Read input table
		logger.info('Reading input data list %s ...' % (self.datalist))

		with open(self.datalist,'r') as f:
		
			for line in f:
				line_split = line.strip().split(',')
				(filename,x1,y1,x2,y2,class_name) = line_split

				if class_name not in self.classes_count:
					self.classes_count[class_name] = 1
				else:
					self.classes_count[class_name] += 1

				if class_name not in self.class_mapping:
					if class_name == 'bg' and self.found_bg == False:
						logger.info('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
						self.found_bg = True
					self.class_mapping[class_name] = len(self.class_mapping)

				if filename not in all_imgs:
					all_imgs[filename] = {}

					# - Read image
					img= self.read_img(filename)
					if img is None:
						logger.error("Failed to read image %s!" % (filename))
						return -1

					
					self.img_height= img.shape[0]
					self.img_width= img.shape[1]	
					if self.C.resize_img:
						(resized_width, resized_height) = Utils.get_new_img_size(self.img_width, self.img_height, self.C.im_size)
						self.img_height= resized_height
						self.img_width= resized_width
					
					if img.ndim==2:
						self.is_gray_level_img= True
						self.img_depth= 1

						gl_chan= img[:,:].flatten()
						r_chan_all= np.concatenate((r_chan_all,gl_chan))
						g_chan_all= np.concatenate((g_chan_all,gl_chan))
						b_chan_all= np.concatenate((b_chan_all,gl_chan))
					elif img.ndim==3:
						self.is_gray_level_img= False
						self.img_depth= 3
						r_chan= img[:,:,0].flatten()
						g_chan= img[:,:,1].flatten()
						b_chan= img[:,:,2].flatten()
						r_chan_all= np.concatenate((r_chan_all,r_chan))
						g_chan_all= np.concatenate((g_chan_all,g_chan))
						b_chan_all= np.concatenate((b_chan_all,b_chan))
					else:
						logger.error("Unknown image dim (%d)!" % (img.dim))
						return -1

					# - Add data to list
					(rows,cols) = img.shape[:2]
					all_imgs[filename]['filepath'] = filename
					all_imgs[filename]['width'] = cols
					all_imgs[filename]['height'] = rows
					all_imgs[filename]['bboxes'] = []

					if self.C.split_train_test_data:
						if np.random.randint(0,6) > 0:
							all_imgs[filename]['imageset'] = 'trainval'
						else:
							all_imgs[filename]['imageset'] = 'test'
					else:
						all_imgs[filename]['imageset'] = 'trainval'					

				all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)), 'y1': int(float(y1)), 'y2': int(float(y2))})
				#print all_imgs[filename]['bboxes']

			self.all_data = []
			for key in all_imgs:
				self.all_data.append(all_imgs[key])
		
			# make sure the bg class is last in the list
			if self.found_bg:
				if self.class_mapping['bg'] != len(self.class_mapping) - 1:
					key_to_switch = [key for key in self.class_mapping.keys() if self.class_mapping[key] == len(self.class_mapping)-1][0]
					val_to_switch = self.class_mapping['bg']
					self.class_mapping['bg'] = len(self.class_mapping) - 1
					self.class_mapping[key_to_switch] = val_to_switch

			# - Added by Simo
			if 'bg' not in self.classes_count:
				self.classes_count['bg'] = 0
				self.class_mapping['bg'] = len(self.class_mapping)
		
			print('Training images per class:')
			pprint.pprint(self.classes_count)
			print('Num classes (including bg) = {}'.format(len(self.classes_count)))

			self.classes = [b for b in self.classes_count.keys() if self.classes_count[b] > 0]
			self.class_cycle = itertools.cycle(self.classes)
			self.curr_class = next(self.class_cycle)

			# - Compute channel means
			self.img_channel_mean[0]= np.mean(r_chan_all)
			self.img_channel_mean[1]= np.mean(g_chan_all)
			self.img_channel_mean[2]= np.mean(b_chan_all)
	

			# - Shuffle data
			random.shuffle(self.all_data)
			num_imgs = len(self.all_data)
			train_imgs = [s for s in self.all_data if s['imageset'] == 'trainval']
			val_imgs = [s for s in self.all_data if s['imageset'] == 'test']
			print('Num train samples {}'.format(len(train_imgs)))
			print('Num val samples {}'.format(len(val_imgs)))

			
			return 0

	#================================
	##     SKIP SAMPLE?
	#================================
	def skip_sample_for_balanced_class(self, img_data):

		class_in_img = False

		for bbox in img_data['bboxes']:

			cls_name = bbox['class']

			if cls_name == self.curr_class:
				class_in_img = True
				self.curr_class = next(self.class_cycle)
				break

		if class_in_img:
			return False
		else:
			return True

	#================================
	##     GET AUGMENTED DATA
	#================================
	def augment(self,img_data, augment=True):
		assert 'filepath' in img_data
		assert 'bboxes' in img_data
		assert 'width' in img_data
		assert 'height' in img_data

		# - Copy input data
		img_data_aug = copy.deepcopy(img_data)

		# - Read image data
		filename= img_data_aug['filepath']
		img= self.read_img(filename)
		if img is None:
			logger.error("Failed to read image %s!" % (filename))
			return None

		# - Get augmented data
		if augment:
			rows, cols = img.shape[:2]

			if self.C.use_horizontal_flips and np.random.randint(0, 2) == 0:
				img = cv2.flip(img, 1)
				for bbox in img_data_aug['bboxes']:
					x1 = bbox['x1']
					x2 = bbox['x2']
					bbox['x2'] = cols - x1
					bbox['x1'] = cols - x2

			if self.C.use_vertical_flips and np.random.randint(0, 2) == 0:
				img = cv2.flip(img, 0)
				for bbox in img_data_aug['bboxes']:
					y1 = bbox['y1']
					y2 = bbox['y2']
					bbox['y2'] = rows - y1
					bbox['y1'] = rows - y2

			if self.C.rot_90:
				angle = np.random.choice([0,90,180,270],1)[0]
				if angle == 270:
					if self.is_gray_level_img:
						img = np.transpose(img)
					else:
						img = np.transpose(img, (1,0,2))
					img = cv2.flip(img, 0)
				elif angle == 180:
					img = cv2.flip(img, -1)
				elif angle == 90:
					if self.is_gray_level_img:
						img = np.transpose(img)
					else:
						img = np.transpose(img, (1,0,2))
					img = cv2.flip(img, 1)
				elif angle == 0:
					pass

				for bbox in img_data_aug['bboxes']:
					x1 = bbox['x1']
					x2 = bbox['x2']
					y1 = bbox['y1']
					y2 = bbox['y2']
					if angle == 270:
						bbox['x1'] = y1
						bbox['x2'] = y2
						bbox['y1'] = cols - x2
						bbox['y2'] = cols - x1
					elif angle == 180:
						bbox['x2'] = cols - x1
						bbox['x1'] = cols - x2
						bbox['y2'] = rows - y1
						bbox['y1'] = rows - y2
					elif angle == 90:
						bbox['x1'] = rows - y2
						bbox['x2'] = rows - y1
						bbox['y1'] = x1
						bbox['y2'] = x2        
					elif angle == 0:
						pass

		img_data_aug['width'] = img.shape[1]
		img_data_aug['height'] = img.shape[0]

		return img_data_aug, img
		
	#================================
	##     GENERATE DATA
	#================================
	def get_anchor_gt(self,shuffle,augment,img_length_calc_function):
		""" Generate data from input """

		# - Get info from data & nn 
		all_img_data= self.all_data
		class_count= self.classes_count	
		
		# - Get data ordering
		backend= K.image_dim_ordering()

		print("Called get_anchor_gt (all_img_data size=%d)" % (len(all_img_data)))

		# - Generate data
		while True:
			if shuffle:
				np.random.shuffle(all_img_data)

			for img_data in all_img_data:
				try:

					
					if self.C.balanced_classes and self.skip_sample_for_balanced_class(img_data):
						logger.warn("Skip class...")
						continue

					out= self.augment(img_data, augment=augment)
					#img_data_aug, x_img = self.augment(img_data, augment=augment)
					if out is None:
						logger.error("Failed to get data, skip to next generation!")
						continue			
					img_data_aug= out[0]
					x_img= out[1]

					(width, height) = (img_data_aug['width'], img_data_aug['height'])
					#(rows, cols, _) = x_img.shape
					x_img_size= x_img.shape
					rows= x_img_size[0]
					cols= x_img_size[1]

					assert cols == width
					assert rows == height

					# - Resize the image so that smallest side is length = 600px by default or desired image size
					(resized_width, resized_height) = width, height
					if self.C.resize_img:
						(resized_width, resized_height) = Utils.get_new_img_size(width, height, self.C.im_size)
						x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

					#print("width=%d, height=%d, resized_width=%d, resized_height=%d" % (width,height,resized_width,resized_height))

					try:
						y_rpn_cls, y_rpn_regr = self.calc_rpn(img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)
					except Exception as e:
						logger.warn("Exception in calc_rpn!")
						print(e)
						continue

					# Zero-center by mean pixel, and preprocess image
					if not self.is_gray_level_img:
						x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
					x_img = x_img.astype(np.float32)
					
					if self.C.subtract_chan_mean:
						if self.is_gray_level_img:
							x_img[:, :] -= self.C.img_channel_mean[0]
						else:
							x_img[:, :, 0] -= self.C.img_channel_mean[0]
							x_img[:, :, 1] -= self.C.img_channel_mean[1]
							x_img[:, :, 2] -= self.C.img_channel_mean[2]

					x_img /= self.C.img_scaling_factor

					if not self.is_gray_level_img:
						x_img = np.transpose(x_img, (2, 0, 1))

					x_img = np.expand_dims(x_img, axis=0)

					y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= self.C.std_scaling

					if backend == 'tf':
						if not self.is_gray_level_img:
							x_img = np.transpose(x_img, (0, 2, 3, 1))
						y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
						y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

					# - Add 4th channel =1 for gray scale images (network expects input of shape (nsamples,ny,nx,nchannels)
					
					if self.is_gray_level_img:
						img_shape= x_img.shape
						x_img= x_img.reshape(img_shape[0],img_shape[1],img_shape[2],1)

					print("== x_img info ==")
					#print(type(x_img))
					print(x_img.shape)	
					print("== y_rpn_cls info ==")
					#print(type(y_rpn_cls))
					print(y_rpn_cls.shape)
					print("== y_rpn_regr info ==")
					#print(type(y_rpn_regr))
					print(y_rpn_regr.shape)

					yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

				except Exception as e:
					print(e)
					continue
	
	#================================
	##     COMPUTE RPN
	#================================
	def calc_rpn(self, img_data, width, height, resized_width, resized_height, img_length_calc_function):
		""" Compute RPN """
	
		downscale = float(self.C.rpn_stride)
		anchor_sizes = self.C.anchor_box_scales
		anchor_ratios = self.C.anchor_box_ratios
		num_anchors = len(anchor_sizes) * len(anchor_ratios)	

		#print("calc_rpn: anchor_ratios")
		#print(anchor_ratios)
		#print("calc_rpn: anchor_sizes")
		#print(anchor_sizes)
		#print("downscale")
		#print(downscale)
		#print("num_anchors")
		#print(num_anchors)

		# calculate the output map size based on the network architecture
		(output_width, output_height) = img_length_calc_function(resized_width, resized_height)

		logger.info("Generating RPN on feature map shape (%d,%d) ..." % (output_width, output_height))

		n_anchratios = len(anchor_ratios)
	
		# initialise empty output objectives
		y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
		y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
		y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

		num_bboxes = len(img_data['bboxes'])

		num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
		best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
		best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
		best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
		best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

		# get the GT box coordinates, and resize to account for image resizing
		gta = np.zeros((num_bboxes, 4))
		for bbox_num, bbox in enumerate(img_data['bboxes']):
			# get the GT box coordinates, and resize to account for image resizing
			gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
			gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
			gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
			gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))
	
		# rpn ground truth

		for anchor_size_idx in range(len(anchor_sizes)):

			for anchor_ratio_idx in range(n_anchratios):
				anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
				anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]	
			
				logger.debug("Generating anchor box size=%d, ratio=[%d,%d], (x,y)=(%d,%d)",anchor_sizes[anchor_size_idx],anchor_ratios[anchor_ratio_idx][0],anchor_ratios[anchor_ratio_idx][1],anchor_x,anchor_y)

	
				for ix in range(output_width):					
					# x-coordinates of the current anchor box	
					x1_anc = downscale * (ix + 0.5) - anchor_x / 2
					x2_anc = downscale * (ix + 0.5) + anchor_x / 2	
				
					# ignore boxes that go across image boundaries					
					if x1_anc < 0 or x2_anc > resized_width:
						continue
					
					for jy in range(output_height):
	
						# y-coordinates of the current anchor box
						y1_anc = downscale * (jy + 0.5) - anchor_y / 2
						y2_anc = downscale * (jy + 0.5) + anchor_y / 2

						# ignore boxes that go across image boundaries
						if y1_anc < 0 or y2_anc > resized_height:
							continue

						#logger.debug("Anchor box (x1,x2,y1,y2)=(%d,%d,%d,%d)",x1_anc,x2_anc,y1_anc,y2_anc)

						# bbox_type indicates whether an anchor should be a target 
						bbox_type = 'neg'

						# this is the best IOU for the (x,y) coord and the current anchor
						# note that this is different from the best IOU for a GT bbox
						best_iou_for_loc = 0.0

						for bbox_num in range(num_bboxes):
						
							# get IOU of the current GT box and the current anchor box
							curr_iou = Utils.iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
							# calculate the regression targets if they will be needed
							if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > self.C.rpn_max_overlap:
								cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
								cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
								cxa = (x1_anc + x2_anc)/2.0
								cya = (y1_anc + y2_anc)/2.0

								tx = (cx - cxa) / (x2_anc - x1_anc)
								ty = (cy - cya) / (y2_anc - y1_anc)
								tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
								th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
						
							if img_data['bboxes'][bbox_num]['class'] != 'bg':

								# all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
								if curr_iou > best_iou_for_bbox[bbox_num]:
									best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
									best_iou_for_bbox[bbox_num] = curr_iou
									best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
									best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]

									logger.debug("Best IOU anchor box (x1,x2,y1,y2)=(%d,%d,%d,%d) for bbox no. %d: IOU=%f",x1_anc,x2_anc,y1_anc,y2_anc,bbox_num,curr_iou)


								# we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
								if curr_iou > self.C.rpn_max_overlap:
									bbox_type = 'pos'
									num_anchors_for_bbox[bbox_num] += 1
				
									logger.debug("Positive anchor box (x1,x2,y1,y2)=(%d,%d,%d,%d), IOU=%f",x1_anc,x2_anc,y1_anc,y2_anc,curr_iou)

									# we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
									if curr_iou > best_iou_for_loc:
										best_iou_for_loc = curr_iou
										best_regr = (tx, ty, tw, th)

								# if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
								if self.C.rpn_min_overlap < curr_iou < self.C.rpn_max_overlap:
									# gray zone between neg and pos
									if bbox_type != 'pos':
										bbox_type = 'neutral'

						# turn on or off outputs depending on IOUs
						if bbox_type == 'neg':
							y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
							y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
						elif bbox_type == 'neutral':
							y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
							y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
						elif bbox_type == 'pos':
							y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
							y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
							start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
							y_rpn_regr[jy, ix, start:start+4] = best_regr

		# we ensure that every bbox has at least one positive RPN region

		for idx in range(num_anchors_for_bbox.shape[0]):
			if num_anchors_for_bbox[idx] == 0:
				# no box with an IOU greater than zero ...
				if best_anchor_for_bbox[idx, 0] == -1:
					continue
				y_is_box_valid[
					best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
					best_anchor_for_bbox[idx,3]] = 1
				y_rpn_overlap[
					best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
					best_anchor_for_bbox[idx,3]] = 1
				start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
				y_rpn_regr[
					best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

			logger.debug("best_anchor_for_bbox[idx,0]=%d" % (best_anchor_for_bbox[idx,0]))

		y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
		y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

		y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
		y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

		y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
		y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

		pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
		neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

		num_pos = len(pos_locs[0])

		# one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
		# regions. We also limit it to 256 regions.
		num_regions = 256

		if len(pos_locs[0]) > num_regions/2:
			val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
			y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
			num_pos = num_regions/2

		if len(neg_locs[0]) + num_pos > num_regions:
			val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
			y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

		#print("y_rpn_regr")
		#print(y_rpn_regr.shape)

		y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
		y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

		return np.copy(y_rpn_cls), np.copy(y_rpn_regr)
	


