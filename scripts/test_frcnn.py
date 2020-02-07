#!/usr/bin/env python

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time

## IMAGE PROC MODULES
import cv2
from keras import backend as K
from keras.layers import Input
from keras.models import Model

## ASTRO MODULES
from astropy.io import ascii

## GRAPHICS MODULES
import matplotlib.pyplot as plt
from matplotlib import patches

## MODULES
from sidelobe_finder import config
from sidelobe_finder import roi_helpers
from sidelobe_finder import data_augment


sys.setrecursionlimit(40000)


###########################
##     OPTIONS
###########################
# - Define options
parser = OptionParser()

parser.add_option("--filename_test", dest="filename_test", help="Test data filename")
#parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",help="Number of ROIs per iteration. Higher means more memory use.", default=32)
#parser.add_option("--config_filename", dest="config_filename", help="Location to read the metadata related to the training (generated when training).",default="config.pickle")
parser.add_option("--filename_config", dest="filename_config", help="Location to read the metadata related to the training (generated when training).",default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
#parser.add_option("--bb_filename", dest="bb_filename", help="Location to read the true bounding box info of test data",default="")
parser.add_option("--anchor_box_scales", dest="anchor_box_scales", help="Anchor box scales", default='2,4,8,16,32')
parser.add_option("--rpn_stride", dest="rpn_stride", type="int", help="RPN stride parameter (Default=16).", default=16)
parser.add_option("--img_size", dest="img_size", type="int", help="Image size per side in pixels (Default=200).", default=200)

# - Parse options
(options, args) = parser.parse_args()

#if not options.test_path:   # if filename is not given
#	parser.error('Error: path to test data must be specified. Pass --path to command line')
if not options.filename_test:   # if filename is not given
	parser.error('Error: path to test data list must be specified. Pass --filename_test to command line')


#config_output_filename = options.config_filename
#bb_filename = options.bb_filename
filename_test= options.filename_test
config_output_filename = options.filename_config
#bb_filename = options.bb_filename

anchor_scales_str= options.anchor_box_scales
anchor_scales_str_list= anchor_scales_str.split(",")
anchor_scales= []
for item in anchor_scales_str_list:
	anchor_scales.append(int(item))

rpn_stride= options.rpn_stride
img_size= options.img_size

# - Read options from stored config and override some with those given from command line
with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

if C.network == 'resnet50':
	import sidelobe_finder.resnet as nn
elif C.network == 'vgg':
	import sidelobe_finder.vgg as nn

# - Turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

#img_path = options.test_path

class_mapping = C.class_mapping

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if C.network == 'resnet50':
	num_features = 1024
elif C.network == 'vgg':
	num_features = 512
else:
	print('Not a valid model')
	raise ValueError

# - Override anchor scales from command line
C.anchor_box_scales= anchor_scales
C.rpn_stride= rpn_stride
C.im_size= img_size

###########################
##     FUNCTIONS
###########################
def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)


###########################
##     MAIN
###########################

#================================
#    READ TRUE BOUNDING BOXES
#================================ 
true_bb_dict= {}
all_imgs = []
#if bb_filename:
#	bb_table= ascii.read(bb_filename)
if filename_test:
	bb_table= ascii.read(filename_test)
	for item in bb_table:
		imgfilename= item['col1']
		x1= item['col2']
		y1= item['col3']
		x2= item['col4']
		y2= item['col5']
		bb_dict= {}
		bb_dict['x1']= x1
		bb_dict['y1']= y1
		bb_dict['x2']= x2
		bb_dict['y2']= y2
		all_imgs.append(imgfilename)
		true_bb_dict[imgfilename]= bb_dict

#================================
#    BUILD NN
#================================ 
# - Initialize NN architecture layout
if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# - Define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# - Define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

# - Define classifier
classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)
model_classifier = Model([feature_map_input, roi_input], classifier)

# - Loading NN weights
print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')


#==================================
#    RUN CLASSIFIER ON TEST DATA
#==================================

classes = {}
bbox_threshold = 0.8
visualise = True

#for idx, img_name in enumerate(sorted(os.listdir(img_path))):
#	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.fits')):
#		continue
#	print(img_name)

for filepath in all_imgs:

	print(filepath)
	st = time.time()
	#filepath = os.path.join(img_path,img_name)
	file_extension = os.path.splitext(filepath)[1]

	# - Read image data
	if file_extension=='.fits':
		img, header= data_augment.read_fits(filepath,stretch=True,normalize=True,convertToRGB=True)
	else:
		img = cv2.imread(filepath)

	# - Read true object bounding boxes (if given)
	if true_bb_dict:
		bb_dict= true_bb_dict[filepath]
		true_bb_x1= bb_dict['x1']
		true_bb_y1= bb_dict['y1']
		true_bb_x2= bb_dict['x2']
		true_bb_y2= bb_dict['y2']	
	
	# - Modify image format
	X, ratio = format_img(img, C)

	if K.image_dim_ordering() == 'tf':
		X = np.transpose(X, (0, 2, 3, 1))

	# - Get the feature maps and output from the RPN
	[Y1, Y2, F] = model_rpn.predict(X)
	
	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)
	
	# - Convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# - Apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}

	for jk in range(R.shape[0]//C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
		if ROIs.shape[1] == 0:
			print("DEBUG: ROIs shape==0, exiting loop!")
			break

		if jk == R.shape[0]//C.num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				continue

			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))

	all_dets = []

	##############################
	##       DRAW FIGURE
	##############################
	# - Create figure & axis
	fig = plt.figure()	
	ax = plt.axes([0,0,1,1], frameon=False)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	plt.autoscale(tight=True)

	# - Draw image
	plt.imshow(img,cmap='gray')

	# - Draw true bounding boxes
	if true_bb_dict:
		true_rect = patches.Rectangle((true_bb_x1,true_bb_y1),true_bb_x2-true_bb_x1,true_bb_y2-true_bb_y1, edgecolor='red', facecolor = 'none')
		ax.add_patch(true_rect)

	# - Draw detected objects
	for key in bboxes:
		bbox = np.array(bboxes[key])

		print("bbox")
		print(bbox)

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]

			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
			width= real_x2 - real_x1
			height= real_y2 - real_y1

			cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

			textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
			all_dets.append((key,100*new_probs[jk]))

			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
			textOrg = (real_x1, real_y1-0)

			cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

			col_r= int(class_to_color[key][0])
			col_g= int(class_to_color[key][1])
			col_b= int(class_to_color[key][2])

			#rect = patches.Rectangle((real_x1,real_y1), width, height, edgecolor = 'yellow', facecolor = 'none')
			rect = patches.Rectangle((real_x1,real_y1), width, height, edgecolor = [(col_r,col_g,col_b)], facecolor = 'none')
			ax.add_patch(rect)
			#plt.scatter(real_x1, real_y1, s=12, color='yellow')
			#plt.scatter(real_x2, real_y2, s=12,color='yellow')
			#ax.annotate(textLabel, xy=(real_x1+0.5*width,real_y1-10),color='yellow')
			plt.scatter(real_x1, real_y1, s=12, color=[(col_r,col_g,col_b)])
			plt.scatter(real_x2, real_y2, s=12,color=[(col_r,col_g,col_b)])
			ax.annotate(textLabel, xy=(real_x1+0.5*width,real_y1-10),color=[(col_r,col_g,col_b)])

	print('Elapsed time = {}'.format(time.time() - st))
	print(all_dets)
	#cv2.imshow('img', img) ## ORIgINAL CODE
	#cv2.waitKey(0) ## ORIGINAL CODE
	# cv2.imwrite('./results_imgs/{}.png'.format(idx),img)
	
	plt.show()



