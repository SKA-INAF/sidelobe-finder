import cv2
import numpy as np
import copy

## ASTRO MODULES
from astropy.io import ascii, fits
from astropy.units import Quantity
from astropy.modeling.parameters import Parameter
from astropy.modeling.core import Fittable2DModel
from astropy import wcs
from astropy import units as u
from astropy.visualization import ZScaleInterval

## ====================================
## ==     ADDED BY SIMO
## ====================================
def read_fits(filename,stretch=True,normalize=True,convertToRGB=True):
	""" Read FITS image """
	
	# - Open file
	try:
		hdu= fits.open(filename,memmap=False)
	except Exception as ex:
		errmsg= 'Cannot read image file: ' + filename
		raise IOError(errmsg)

	# - Read data
	data= hdu[0].data
	data_size= np.shape(data)
	nchan= len(data.shape)
	if nchan==4:
		output_data= data[0,0,:,:]
	elif nchan==2:
		output_data= data	
	else:
		errmsg= 'Invalid/unsupported number of channels found in file ' + filename + ' (nchan=' + str(nchan) + ')!'
		hdu.close()
		raise IOError(errmsg)

	# - Convert data to float 32
	output_data= output_data.astype(np.float32)

	# - Read metadata
	header= hdu[0].header

	# - Close file
	hdu.close()

	#print("Fits file shape (%d,%d)" % (output_data.shape[0],output_data.shape[1]))

	# - Replace nan values with min pix value
	img_min= np.nanmin(output_data)
	output_data[np.isnan(output_data)]= img_min	

	# - Stretch data using zscale transform
	if stretch:
		data_stretched= stretch_img(output_data)
		output_data= data_stretched
		output_data= output_data.astype(np.float32)

	# - Normalize data to [0,255]
	if normalize:
		data_norm= normalize_img(output_data)
		output_data= data_norm
		output_data= output_data.astype(np.float32)

	# - Convert to RGB image
	if convertToRGB:
		if not normalize:
			data_norm= normalize_img(output_data)
			output_data= data_norm
		data_rgb= gray2rgb(output_data) 
		output_data= data_rgb

	return output_data, header
	

def stretch_img(data,contrast=0.25):
	""" Apply z-scale stretch to image """
	#print("Apply z-scale stretch to image ...")
	
	transform= ZScaleInterval(contrast=contrast)
	data_stretched= transform(data)
	#print("Apply z-scale stretch to image ... done!")
	
	return data_stretched


def normalize_img(data):
	""" Normalize image to (0,1) """
	#print("Normalize image ...")
	#data_info = np.iinfo(data.dtype)
	#print(data_info)
	#data_norm= data.astype(np.float32) / data_info.max # normalize the data to 0 - 1
	#data_norm= 255 * data_norm # Now scale by 255

	data_max= np.max(data)
	#data_norm= 255* data/data_max
	data_norm= data/data_max

	return data_norm

def gray2rgb(data_float):
	""" Convert gray image data to rgb """

	# - Convert to uint8
	data_uint8 = np.array( (data_float*255).round(), dtype = np.uint8)
	#print("data min/max=%d/%d" % (data_uint8.min(),data_uint8.max()))

	# - Convert to uint8 3D
	data3_uint8 = np.stack((data_uint8,)*3, axis=-1)

	return data3_uint8

## ====================================

def augment(img_data, config, augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)

	## ====================================
	## ==     ADDED BY SIMO
	## ====================================
	##img = cv2.imread(img_data_aug['filepath']) ## ORIGINAL CODE
	#print("Read FITS ...")
	filename= img_data_aug['filepath']
	img, header= read_fits(filename,stretch=True,normalize=True,convertToRGB=True)
	#print(img.shape)
	#print("==> DATA READ DONE")
	############################################

	if augment:
		rows, cols = img.shape[:2]

		if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 1)
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2

		if config.use_vertical_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 0)
			for bbox in img_data_aug['bboxes']:
				y1 = bbox['y1']
				y2 = bbox['y2']
				bbox['y2'] = rows - y1
				bbox['y1'] = rows - y2

		if config.rot_90:
			angle = np.random.choice([0,90,180,270],1)[0]
			if angle == 270:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 0)
			elif angle == 180:
				img = cv2.flip(img, -1)
			elif angle == 90:
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
