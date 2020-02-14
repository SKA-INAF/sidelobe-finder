##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import os
import sys
import string
import logging
import io
import math

from keras import backend as K

import ConfigParser


##############################
##     GLOBAL VARS
##############################
logger = logging.getLogger(__name__)

###########################
##     CLASS DEFINITIONS
###########################

class Config(object):
	
	""" Configuration options """

	def __init__(self):

		# - Config options
		self.__set_defaults()

		# - Config file parser		
		self.parser= ConfigParser.ConfigParser()

	#==============================
	#     SET DEFAULT OPTIONS
	#==============================
	def __set_defaults(self):
		""" Set default options """

		self.verbose = True

		# - Network
		self.network = 'resnet50'
		self.load_weights= True
		self.network_weights= ''

		# - User network architecture pars
		self.conv_net_arc_file = '' 
		self.conv_layers_types= ['Conv2D','MaxPooling2D','Conv2D','MaxPooling2D','Conv2D']
		self.nfilters_conv= [32,0,64,0,128]
		self.kern_size_conv= [3,2,3,2,3]
		self.stride_conv= [3,2,3,2,3]
		self.rpn_layers_types= ['Conv2D']
		self.nfilters_rpn= [128]
		self.kern_size_rpn= [3]
		self.pooling_region_size = 7
		self.nlayers_class= 2
		self.class_dense_layer_size= 4096
		self.use_dropout= True
		self.dropout= 0.5

		# - Training
		self.nepochs = 10
		self.epoch_length= 1000
		self.learning_rate= 1.e-5
		self.loss_outfile= 'nn_loss.dat'
		self.weight_outfile= 'nn_weights.h5'

		# - Data management and augmentation
		self.split_train_test_data= False
		self.apply_zscale= True
		self.normalize_data= True
		self.convert_to_rgb= True
		self.augment_data= True
		self.use_horizontal_flips = True 
		self.use_vertical_flips = True
		self.rot_90 = True
		self.shuffle_data= True
	
		# size to resize the smallest side of the image
		self.resize_img= False
		#self.im_size = 600
		#self.im_size = 51
		self.im_size = 200

		# image channel-wise mean to subtract
		self.subtract_chan_mean= True
		#self.img_channel_mean = [103.939, 116.779, 123.68]
		self.img_channel_mean = [112,112,112]
		self.img_scaling_factor = 1.0

		# - Anchor boxes
		# anchor box scales
		#self.anchor_box_scales = [128, 256, 512]
		#self.anchor_box_scales = [8, 16, 32]
		self.anchor_box_scales = [2, 4, 8, 16, 32]

		# anchor box ratios
		#self.anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]
		self.anchor_box_ratios = [ [1,1], [1,2], [2,1], [1,3], [3,1] ]

		# stride at the RPN (this depends on the network configuration)
		self.rpn_stride = 16

		# number of ROIs at once
		self.num_rois = 4

		
		self.balanced_classes = False

		# scaling the stdev
		self.std_scaling = 4.0
		self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

		# overlaps for RPN
		self.rpn_min_overlap = 0.3
		self.rpn_max_overlap = 0.7

		# overlaps for classifier ROIs
		self.classifier_min_overlap = 0.1
		self.classifier_max_overlap = 0.5

		# placeholder for the class mapping, automatically generated by the parser
		self.class_mapping = None

		#location of pretrained weights for the base network 
		# weight files can be found at:
		# https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5
		# https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

		#self.model_path = 'model_frcnn.vgg.hdf5'
		self.model_path = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

	#==============================
	#     PARSE CONFIG FILE
	#==============================
	def parse(self,filename):
		""" Read input INI config file and set options """

		# ***************************
		# **    READ CONFIG FILE
		# ***************************
		# - Read config parser
		self.parser.read(filename)

		# ***************************
		# **    PARSE OPTIONS
		# ***************************
		# - Parse DATA section options
		if self.parser.has_option('DATA_PREPROCESSING', 'split_train_test_data'):
			self.split_train_test_data= self.parser.getboolean('DATA_PREPROCESSING', 'split_train_test_data')

		if self.parser.has_option('DATA_PREPROCESSING', 'apply_zscale'):
			self.apply_zscale= self.parser.getboolean('DATA_PREPROCESSING', 'apply_zscale')				

		if self.parser.has_option('DATA_PREPROCESSING', 'normalize_data'):
			self.normalize_data= self.parser.getboolean('DATA_PREPROCESSING', 'normalize_data')				

		if self.parser.has_option('DATA_PREPROCESSING', 'convert_to_rgb'):
			self.convert_to_rgb= self.parser.getboolean('DATA_PREPROCESSING', 'convert_to_rgb')				

		if self.parser.has_option('DATA_PREPROCESSING', 'augment_data'):
			self.augment_data= self.parser.getboolean('DATA_PREPROCESSING', 'augment_data')	

		if self.parser.has_option('DATA_PREPROCESSING', 'use_horizontal_flips'):
			self.use_horizontal_flips= self.parser.getboolean('DATA_PREPROCESSING', 'use_horizontal_flips')				

		if self.parser.has_option('DATA_PREPROCESSING', 'use_vertical_flips'):
			self.use_vertical_flips= self.parser.getboolean('DATA_PREPROCESSING', 'use_vertical_flips')				

		if self.parser.has_option('DATA_PREPROCESSING', 'use_90deg_rotation'):
			self.rot_90= self.parser.getboolean('DATA_PREPROCESSING', 'use_90deg_rotation')	

		if self.parser.has_option('DATA_PREPROCESSING', 'resize_img'):
			self.resize_img= self.parser.getboolean('DATA_PREPROCESSING', 'resize_img')	

		if self.parser.has_option('DATA_PREPROCESSING', 'img_size'):
			option_value= self.parser.get('DATA_PREPROCESSING', 'img_size')
			if option_value:
				self.im_size= int(option_value)

		if self.parser.has_option('DATA_PREPROCESSING', 'subtract_chan_mean'):
			self.subtract_chan_mean= self.parser.getboolean('DATA_PREPROCESSING', 'subtract_chan_mean')	
		
		# - Parse NETWORK section options
		if self.parser.has_option('NETWORK', 'network'):
			option_value= self.parser.get('NETWORK', 'network')	
			if option_value:
				self.network= option_value

		if self.parser.has_option('NETWORK', 'load_weights'):
			self.load_weights= self.parser.getboolean('NETWORK', 'load_weights')	

		if self.parser.has_option('NETWORK', 'weights'):
			option_value= self.parser.get('NETWORK', 'weights')	
			if option_value:
				self.network_weights= option_value
		
		if self.parser.has_option('NETWORK', 'num_rois'):
			option_value= self.parser.get('NETWORK', 'num_rois')
			if option_value:
				self.num_rois= int(option_value)


		# - Parse USER_NETWORK section options
		if self.parser.has_option('USER_NETWORK', 'conv_layers_types'):
			option_value= self.parser.get('USER_NETWORK', 'conv_layers_types')	
			if option_value:
				self.conv_layers_types= option_value.split(",")

		if self.parser.has_option('USER_NETWORK', 'nfilters_conv'):
			option_value= self.parser.get('USER_NETWORK', 'nfilters_conv')	
			if option_value:
				options_list= option_value.split(",")
				self.nfilters_conv= []
				for opt in options_list:
					self.nfilters_conv.append(int(opt))

				
		if self.parser.has_option('USER_NETWORK', 'kern_size_conv'):
			option_value= self.parser.get('USER_NETWORK', 'kern_size_conv')	
			if option_value:
				options_list= option_value.split(",")
				self.kern_size_conv= []
				for opt in options_list:
					self.kern_size_conv.append(int(opt))

		if self.parser.has_option('USER_NETWORK', 'stride_conv'):
			option_value= self.parser.get('USER_NETWORK', 'stride_conv')	
			if option_value:
				options_list= option_value.split(",")
				self.stride_conv= []
				for opt in options_list:
					self.stride_conv.append(int(opt))


		if self.parser.has_option('USER_NETWORK', 'rpn_layers_types'):
			option_value= self.parser.get('USER_NETWORK', 'rpn_layers_types')	
			if option_value:
				self.rpn_layers_types= option_value.split(",")

		if self.parser.has_option('NETWORK', 'nfilters_rpn'):
			option_value= self.parser.get('NETWORK', 'nfilters_rpn')	
			if option_value:
				options_list= option_value.split(",")
				self.nfilters_rpn= []
				for opt in options_list:
					self.nfilters_rpn.append(int(opt))

		if self.parser.has_option('USER_NETWORK', 'kern_size_rpn'):
			option_value= self.parser.get('USER_NETWORK', 'kern_size_rpn')	
			if option_value:
				options_list= option_value.split(",")
				self.kern_size_rpn= []
				for opt in options_list:
					self.kern_size_rpn.append(int(opt))

		if self.parser.has_option('USER_NETWORK', 'pooling_region_size'):
			option_value= self.parser.get('USER_NETWORK', 'pooling_region_size')
			if option_value:
				self.pooling_region_size= int(option_value)

		if self.parser.has_option('USER_NETWORK', 'nlayers_class'):
			option_value= self.parser.get('USER_NETWORK', 'nlayers_class')
			if option_value:
				self.nlayers_class= int(option_value)

		if self.parser.has_option('USER_NETWORK', 'class_dense_layer_size'):
			option_value= self.parser.get('USER_NETWORK', 'class_dense_layer_size')
			if option_value:
				self.class_dense_layer_size= int(option_value)

		if self.parser.has_option('USER_NETWORK', 'use_dropout'):
			self.use_dropout= self.parser.getboolean('USER_NETWORK', 'use_dropout')	
		
		if self.parser.has_option('USER_NETWORK', 'dropout'):
			option_value= self.parser.get('USER_NETWORK', 'dropout')
			if option_value:
				self.dropout= float(option_value)

		# - Parse ANCHOR section options
		if self.parser.has_option('ANCHOR_BOXES', 'rpn_stride'):
			option_value= self.parser.get('ANCHOR_BOXES', 'rpn_stride')
			if option_value:
				self.rpn_stride= int(option_value)
		
		if self.parser.has_option('ANCHOR_BOXES', 'anchor_box_scales'):
			option_value= self.parser.get('ANCHOR_BOXES', 'anchor_box_scales')	
			if option_value:
				options_list= option_value.split(",")
				self.anchor_box_scales= []
				for opt in options_list:
					self.anchor_box_scales.append(int(opt))
	
		if self.parser.has_option('ANCHOR_BOXES', 'force_rectangular_anchor'):
			self.force_rectangular_anchor= self.parser.getboolean('ANCHOR_BOXES', 'force_rectangular_anchor')	
		
		if self.parser.has_option('ANCHOR_BOXES', 'anchor_box_ratios'):
			option_value= self.parser.get('ANCHOR_BOXES', 'anchor_box_ratios')	
			if option_value:
				options_list= option_value.split(",")
				self.anchor_box_ratios= []
				n_ratios= len(options_list)
				for index in range(n_ratios):
					r= options_list[index]
					r0= options_list[0]

					if r==r0:
						if not self.force_rectangular_anchor:	
							self.anchor_box_ratios.append([int(r),int(r0)])
					else:
						self.anchor_box_ratios.append([int(r),int(r0)])
						self.anchor_box_ratios.append([int(r0),int(r)])				
	
		if self.parser.has_option('ANCHOR_BOXES', 'rpn_min_overlap'):
			option_value= self.parser.get('ANCHOR_BOXES', 'rpn_min_overlap')
			if option_value:
				self.rpn_min_overlap= float(option_value)

		if self.parser.has_option('ANCHOR_BOXES', 'rpn_max_overlap'):
			option_value= self.parser.get('ANCHOR_BOXES', 'rpn_max_overlap')
			if option_value:
				self.rpn_max_overlap= float(option_value)
	

		# - Parse TRAIN section options
		if self.parser.has_option('TRAIN', 'nepochs'):
			option_value= self.parser.get('TRAIN', 'nepochs')
			if option_value:
				self.nepochs= int(option_value)

		if self.parser.has_option('TRAIN', 'epoch_length'):
			option_value= self.parser.get('TRAIN', 'epoch_length')
			if option_value:
				self.epoch_length= int(option_value)

		if self.parser.has_option('TRAIN', 'learning_rate'):
			option_value= self.parser.get('TRAIN', 'learning_rate')
			if option_value:
				self.learning_rate= float(option_value)

		if self.parser.has_option('TRAIN', 'loss_outfile'):
			option_value= self.parser.get('TRAIN', 'loss_outfile')
			if option_value:
				self.loss_outfile= option_value
		
		if self.parser.has_option('TRAIN', 'weight_outfile'):
			option_value= self.parser.get('TRAIN', 'weight_outfile')
			if option_value:
				self.weight_outfile= option_value


		return 0

