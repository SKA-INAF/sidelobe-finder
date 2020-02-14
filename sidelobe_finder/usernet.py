#!/usr/bin/env python

from __future__ import print_function
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

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

## KERAS MODULES
#import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.utils import plot_model
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
#from keras.layers.core import Activation
#from keras.layers.core import Dropout
#from keras.layers.core import Lambda
#from keras.layers.core import Dense
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.utils.generic_utils import get_custom_objects
from keras.engine.topology import get_source_inputs
from keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf

## MODULES
from sidelobe_finder.RoiPoolingConv import RoiPoolingConv
from sidelobe_finder import losses as losses

##############################
##     GLOBAL VARS
##############################
logger = logging.getLogger(__name__)


##############################
##     NetworkModel CLASS
##############################
class NetworkModel(object):
	""" Class to create a neural net """
	
	def __init__(self,data_provider,config):
		""" Return a Network object """

		self.dp= data_provider
		self.C= config
		self.img_input= None
		self.roi_input= None
		self.conv_layer= None
		self.conv_net_tot_stride= 1
		self.x_class= None
		self.x_regr= None
		self.out_class= None
		self.out_regr= None
		self.model_rpn= None
		self.model_classifier= None
		self.model= None

	def get_input_shape(self):
		""" Compute input shape from DataProvider info """

		if self.dp.is_gray_level_img:
			if K.image_dim_ordering() == 'th':
				#input_shape = (1,None,None) # accept arbitrary shapes
				input_shape = (1,self.dp.img_height,self.dp.img_width)
			else:
				#input_shape = (None,None,1) # accept arbitrary shapes
				input_shape = (self.dp.img_height,self.dp.img_width,1)
				
		else:
			if K.image_dim_ordering() == 'th':
				#input_shape = (3,None,None)		
				input_shape = (self.dp.img_depth,self.dp.img_height,self.dp.img_width)					
			else:
				#input_shape = (None,None,3)	
				input_shape = (self.dp.img_height,self.dp.img_width,self.dp.img_depth)	
				
		return input_shape

	def get_img_output_length(self,width, height):
		def get_output_length(input_length):
			return input_length//self.conv_net_tot_stride

		return get_output_length(width), get_output_length(height)    


	#================================
	##     BUILD CONV NN
	#================================
	def build_conv_nn(self):
		""" Build CNN feature network """
	
		# - Input layer
		inputShape = self.get_input_shape()
		print ("Input shape ")
		print(inputShape)
		self.img_input= Input(shape=inputShape, dtype='float', name='input')
		x= self.img_input

		# - Create conv layers
		stride_prod= 1
		conv_layer_counter= 0
		maxpool_layer_counter= 0


		# Block 1
		#x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
		#x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
		#x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

		# Block 2
		#x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
		#x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
		#x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

		# Block 3
		#x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
		#x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
		#x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
		#x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

		# Block 4
		#x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
		#x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
		#x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
		#x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

		# Block 5
		#x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
		#x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
		#x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
		# x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
		#stride_prod= 16
		
		for index, layer_type in enumerate(self.C.conv_layers_types):
      
			if layer_type=='Conv2D':
				conv_layer_counter+= 1
				name='conv_' + str(conv_layer_counter)
				nfilters= self.C.nfilters_conv[index]
				kernSize= self.C.kern_size_conv[index]
				stride= self.C.stride_conv[index]
				activation= 'relu'
				padding= 'same'
				x = Conv2D(filters=nfilters, kernel_size=(kernSize,kernSize), strides=(stride,stride), activation=activation, padding=padding, name=name)(x)			
	
			elif layer_type=='MaxPooling2D':
				maxpool_layer_counter+= 1
				name='maxpool_' + str(maxpool_layer_counter)
				poolSize= self.C.kern_size_conv[index]
				#padding= 'valid'
				padding= 'same'
				stride= self.C.stride_conv[index]
				stride_prod*= stride
				x = MaxPooling2D(pool_size=(poolSize,poolSize),strides=(stride,stride),padding=padding,name=name)(x)

			else:
				logger.error("Invalid/unknown layer type parsed (%s)!" % layer_type)
				return -1

		self.conv_layer= x
		self.conv_net_tot_stride= stride_prod
		logger.info("Conv net total stride=%d" % (self.conv_net_tot_stride))

		return 0


	
	#================================
	##     BUILD RPN NETWORK
	#================================
	def build_rpn_nn(self,num_anchors):
		""" Build RPN network """

		x= self.conv_layer
		rpn_layer_counter= 0

		for index, layer_type in enumerate(self.C.rpn_layers_types):
      
			if layer_type=='Conv2D':
				rpn_layer_counter+= 1
				name='rpn_' + str(rpn_layer_counter)
				nfilters= self.C.nfilters_rpn[index]
				kernSize= self.C.kern_size_rpn[index]
				stride= 1
				activation= 'relu'
				padding= 'same'
				x = layers.Conv2D(
					filters=nfilters, 
					kernel_size=(kernSize,kernSize), 
					strides=(stride,stride), 
					activation=activation, 
					padding=padding, 
					kernel_initializer='normal',
					name=name)(x)			
				
			else:
				logger.error("Invalid/unknown layer type parsed (%s)!" % layer_type)
				return -1

		#x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(self.conv_layer)

		self.x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
		self.x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

		return 0

	#================================
	##     BUILD CLASSIFIER NETWORK
	#================================
	def build_classifier_nn(self,num_rois, nb_classes = 21):
		""" Build classifier network """

		self.roi_input= Input(shape=(None, 4))
		
		pooling_regions = self.C.pooling_region_size

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
		#if K.backend() == 'tensorflow':
		#	pooling_regions = 7
		#	input_shape = (num_rois,7,7,512)
		#elif K.backend() == 'theano':
		#	pooling_regions = 7
		#	input_shape = (num_rois,512,7,7)

		out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([self.conv_layer, self.roi_input])

		out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)

		for index in range(self.C.nlayers_class):
			name='fc_' + str(index+1)
			activation= 'relu'
			out = TimeDistributed(Dense(self.C.class_dense_layer_size, activation=activation, name=name))(out)
			if self.C.use_dropout:
				out = TimeDistributed(Dropout(self.C.dropout))(out)

		#out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
		#out = TimeDistributed(Dropout(0.5))(out)
		#out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
		#out = TimeDistributed(Dropout(0.5))(out)

		self.out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
		# note: no regression target for bg class
		self.out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

		return 0

	#================================
	##     BUILD NETWORK
	#================================
	def build_network(self):
		""" Build NN """

		# - Define the base network 
		logger.info("Building base conv net ...")
		status= self.build_conv_nn()
		if status<0:
			logger.error("Failed to build base conv net!")
			return -1

		# - Define the RPN network
		num_anchors = len(self.C.anchor_box_scales) * len(self.C.anchor_box_ratios)
		logger.info("Building RPN network using #%d anchors ..." % (num_anchors))
		status= self.build_rpn_nn(num_anchors)
		if status<0:
			logger.error("Failed to build RPN net!")
			return -1

		# - Define the classifier network
		classes_count= self.dp.classes_count
		status= self.build_classifier_nn(self.C.num_rois, nb_classes=len(classes_count))
		if status<0:
			logger.error("Failed to build classifier net!")
			return -1

		# - Define models
		print("img_input shape")
		print(self.img_input.shape)
		print("self.x_class shape")
		print(self.x_regr.shape)
		print("self.x_regr shape")
		print(self.x_regr.shape)

		self.model_rpn = Model(self.img_input, [self.x_class,self.x_regr])
		self.model_classifier = Model([self.img_input, self.roi_input], [self.out_class,self.out_regr])

		# This is a model that holds both the RPN and the classifier, used to load/save weights for the models
		self.model = Model(
			[self.img_input,self.roi_input], 
			[self.x_class,self.x_regr] + [self.out_class,self.out_regr],
			name="SourceDetectionNet"
		)

		# - Print network architecture
		self.model.summary()

		# - Load model weights
		if self.C.load_weights and self.C.network_weights:
			try:
				logger.info("Loading weights from %s " % (self.C.base_net_weights))
				self.model_rpn.load_weights(self.C.weights, by_name=True)
				self.model_classifier.load_weights(self.C.weights, by_name=True)
			except:
				logger.info("Could not load pre-trained model weights %s, using randomized weights ..." % (self.C.weights))

		# - Set training algorithm
		optimizer = Adam(lr=self.C.learning_rate)
		optimizer_classifier = Adam(lr=self.C.learning_rate)
		
		# - Compile model
		self.model_rpn.compile(
			optimizer=optimizer, 
			loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)]
		)

		self.model_classifier.compile(
			optimizer=optimizer_classifier, 
			loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], 
			metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'}
		)

		self.model.compile(optimizer='sgd', loss='mae')



		return 0


