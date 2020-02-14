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


## IMAGE PROC MODULES
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras.utils import generic_utils


## PACKAGE MODULES
#from .utils import Utils
#from .data_provider import DataProvider
import sidelobe_finder.roi_helpers as roi_helpers

##############################
##     GLOBAL VARS
##############################
logger = logging.getLogger(__name__)


##############################
##     CLASS DEFINITIONS
##############################
class NNTrainer(object):
	""" Class to train network """

	def __init__(self,data_provider,nn,config):
		""" Return a DataProvider object """

		self.nn= nn
		self.dp= data_provider
		self.C= config

		# - Options
		self.balanced_classes = False
		
	#================================
	##     RUN TRAIN
	#================================
	def run_train(self):
		""" Run training """
	
		# ***************************
		#  Compute proposed boxes
		# ***************************
		logger.info("Computing proposed bounding boxes ...")
		img_length_calc_function= self.nn.get_img_output_length
		print(type(img_length_calc_function))

		data_gen_train= self.dp.get_anchor_gt(self.C.shuffle_data,self.C.augment_data,img_length_calc_function)

		# ***************************
		#  Train loop
		# ***************************
		logger.info("Start training (#nepochs=%d, epoch_length=%d) ..." % (self.C.nepochs,self.C.epoch_length))
		losses = np.zeros((self.C.epoch_length, 5))
		rpn_accuracy_rpn_monitor = []
		rpn_accuracy_for_epoch = []	
		iter_num = 0
		best_loss = np.Inf
		start_time = time.time()

		for epoch_num in range(self.C.nepochs):

			progbar = generic_utils.Progbar(self.C.epoch_length)
			logger.info('Epoch {}/{}'.format(epoch_num + 1, self.C.nepochs))

			while True:
				try:

					if len(rpn_accuracy_rpn_monitor) == self.C.epoch_length:
						mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
						rpn_accuracy_rpn_monitor = []
						logger.info('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, self.C.epoch_length))
						if mean_overlapping_bboxes == 0:
							logger.warn('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

					X, Y, img_data = next(data_gen_train)

					#print("X shape")
					#print(X.shape)
					#print("Y[0] shape")
					#print(Y[0].shape)
					#print("Y[1] shape")
					#print(Y[1].shape)

					#print("pto 1")
					loss_rpn = self.nn.model_rpn.train_on_batch(X, Y)
					#print("pto 2")
					P_rpn = self.nn.model_rpn.predict_on_batch(X)
					#print("pto 3")

					R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1],self.C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
			
					#print("pto 4")

					# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
					X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, self.C, self.dp.class_mapping)

					#print("pto 5")

					if X2 is None:
						logger.warn("X2 is none (loss_rpn=%f,%f), skip to next data... " % (loss_rpn[0],loss_rpn[1]))
						rpn_accuracy_rpn_monitor.append(0)
						rpn_accuracy_for_epoch.append(0)
						continue

					neg_samples = np.where(Y1[0, :, -1] == 1)
					pos_samples = np.where(Y1[0, :, -1] == 0)

					if len(neg_samples) > 0:
						neg_samples = neg_samples[0]
					else:
						neg_samples = []

					if len(pos_samples) > 0:
						pos_samples = pos_samples[0]
					else:
						pos_samples = []
			
					rpn_accuracy_rpn_monitor.append(len(pos_samples))
					rpn_accuracy_for_epoch.append((len(pos_samples)))

					if self.C.num_rois > 1:
						if len(pos_samples) < self.C.num_rois//2:
							selected_pos_samples = pos_samples.tolist()
						else:
							selected_pos_samples = np.random.choice(pos_samples, self.C.num_rois//2, replace=False).tolist()
						try:
							selected_neg_samples = np.random.choice(neg_samples, self.C.num_rois - len(selected_pos_samples), replace=False).tolist()
						#except:
						#	selected_neg_samples = np.random.choice(neg_samples, self.C.num_rois - len(selected_pos_samples), replace=True).tolist()
	
						#================================
						#=      ADDED BY SIMO
						#================================	
						# Added according to https://github.com/kbardool/keras-frcnn/issues/21
						except ValueError:
							try:
								selected_neg_samples = np.random.choice(neg_samples, self.C.num_rois - len(selected_pos_samples), replace=True).tolist()
							except:
								# The neg_samples is [[1 0 ]] only, therefore there's no negative sample
								continue
						#=================================

						sel_samples = selected_pos_samples + selected_neg_samples
					else:
						# in the extreme case where num_rois = 1, we pick a random pos or neg sample
						selected_pos_samples = pos_samples.tolist()
						selected_neg_samples = neg_samples.tolist()
						if np.random.randint(0, 2):
							sel_samples = random.choice(neg_samples)
						else:
							sel_samples = random.choice(pos_samples)


					loss_class = self.nn.model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

					losses[iter_num, 0] = loss_rpn[1]
					losses[iter_num, 1] = loss_rpn[2]

					losses[iter_num, 2] = loss_class[1]
					losses[iter_num, 3] = loss_class[2]
					losses[iter_num, 4] = loss_class[3]

					progbar.update(
						iter_num+1, 
						[('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),('detector_cls', losses[iter_num, 2]), ('detector_regr', losses[iter_num, 3])]
					)

					iter_num += 1

					if iter_num == self.C.epoch_length:
					
						loss_rpn_cls = np.mean(losses[:, 0])
						loss_rpn_regr = np.mean(losses[:, 1])
						loss_class_cls = np.mean(losses[:, 2])
						loss_class_regr = np.mean(losses[:, 3])
						class_acc = np.mean(losses[:, 4])

						mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
						rpn_accuracy_for_epoch = []

						logger.info('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
						logger.info('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
						logger.info('Loss RPN classifier: {}'.format(loss_rpn_cls))
						logger.info('Loss RPN regression: {}'.format(loss_rpn_regr))
						logger.info('Loss Detector classifier: {}'.format(loss_class_cls))
						logger.info('Loss Detector regression: {}'.format(loss_class_regr))
						logger.info('Elapsed time: {}'.format(time.time() - start_time))

						curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr

						# ==============================
						# ===   ADDED BY SIMO
						#	==============================
						with open(self.C.loss_outfile, "a") as loss_file:
							loss_file.write("%d %f %f %f %f %f %f\n" % (epoch_num,loss_rpn_cls,loss_rpn_regr,loss_class_cls,loss_class_regr,curr_loss,class_acc))	
						# ==============================

						iter_num = 0

						start_time = time.time()

						if curr_loss < best_loss:
							logger.info('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
							best_loss = curr_loss
							self.nn.model.save_weights(self.C.weight_outfile)

						break

				except Exception as e:
					logger.info('Exception: {}'.format(e))
					continue		

		logger.info('Training complete, exiting.')

		return 0

