#!/usr/bin/env python

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import os
import sys
import string
import logging
import numpy as np

## ASTRO MODULES
from astropy.io import ascii, fits
from astropy.units import Quantity
from astropy.modeling.parameters import Parameter
from astropy.modeling.core import Fittable2DModel
from astropy import wcs
from astropy import units as u
from astropy.visualization import ZScaleInterval

## GRAPHICS MODULES
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


###########################
##     CLASS DEFINITIONS
###########################
class Utils(object):
	""" Class collecting utility methods

			Attributes:
				None
	"""

	def __init__(self):
		""" Return a Utils object """
		
	@classmethod
	def get_base_filename(cls,filename):
		""" Get base filename without extension """
		filename_base= os.path.basename(filename)
		return filename_base

	@classmethod
	def get_base_filename_noext(cls,filename):
		""" Get base filename without extension """
		filename_base= os.path.basename(filename)
		filename_base_noext= os.path.splitext(filename_base)[0]
		return filename_base_noext

	@classmethod
	def get_filename_ext(cls,filename):
		""" Get filename extension """
		filename_ext= os.path.splitext(filename)[1]
		return filename_ext

	@classmethod
	def read_ascii(cls,filename,skip_patterns=[]):
		""" Read an ascii file line by line """
	
		try:
			f = open(filename, 'r')
		except IOError:
			errmsg= 'Could not read file: ' + filename
			logger.error(errmsg)
			raise IOError(errmsg)

		fields= []
		for line in f:
			line = line.strip()
			line_fields = line.split()
			if not line_fields:
				continue

			# Skip pattern
			skipline= cls.has_patterns_in_string(line_fields[0],skip_patterns)
			if skipline:
				continue 		

			fields.append(line_fields)

		f.close()	

		return fields

	@classmethod
	def read_fits(cls,filename,stretch=True,normalize=True,convertToRGB=True):
		""" Read FITS image """
	
		# - Open file
		try:
			hdu= fits.open(filename,memmap=False)
		except Exception as ex:
			errmsg= 'Cannot read image file: ' + filename
			logger.error(errmsg)
			return None

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
			logger.error(errmsg)
			return None

		# - Convert data to float 32
		output_data= output_data.astype(np.float32)

		# - Read metadata
		header= hdu[0].header

		# - Close file
		hdu.close()

		# - Replace nan values with min pix value
		img_min= np.nanmin(output_data)
		output_data[np.isnan(output_data)]= img_min	

		# - Stretch data using zscale transform
		if stretch:
			data_stretched= Utils.stretch_img(output_data)
			output_data= data_stretched
			output_data= output_data.astype(np.float32)

		# - Normalize data to [0,255]
		if normalize:
			data_norm= Utils.normalize_img(output_data)
			output_data= data_norm
			output_data= output_data.astype(np.float32)

		# - Convert to RGB image
		if convertToRGB:
			if not normalize:
				data_norm= Utils.normalize_img(output_data)
				output_data= data_norm
			data_rgb= Utils.gray2rgb(output_data) 
			output_data= data_rgb

		return output_data, header
	
	@classmethod
	def stretch_img(cls,data,contrast=0.25):
		""" Apply z-scale stretch to image """
		
		transform= ZScaleInterval(contrast=contrast)
		data_stretched= transform(data)
	
		return data_stretched

	@classmethod
	def normalize_img(cls,data):
		""" Normalize image to (0,1) """
	
		data_max= np.max(data)
		data_norm= data/data_max

		return data_norm

	@classmethod
	def gray2rgb(cls,data_float):
		""" Convert gray image data to rgb """

		# - Convert to uint8
		data_uint8 = np.array( (data_float*255).round(), dtype = np.uint8)
	
		# - Convert to uint8 3D
		data3_uint8 = np.stack((data_uint8,)*3, axis=-1)

		return data3_uint8

	@classmethod
	def union(cls,au, bu, area_intersection):
		""" Compute union of bounding boxes """
		area_a = (au[2] - au[0]) * (au[3] - au[1])
		area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
		area_union = area_a + area_b - area_intersection
		return area_union

	@classmethod
	def intersection(cls,ai, bi):
		""" Compute intersection of bounding boxes """
		x = max(ai[0], bi[0])
		y = max(ai[1], bi[1])
		w = min(ai[2], bi[2]) - x
		h = min(ai[3], bi[3]) - y
		if w < 0 or h < 0:
			return 0
		return w*h

	@classmethod
	def iou(cls,a,b):
		""" Compute IOU of bounding boxes"""

		# a and b should be (x1,y1,x2,y2)
		if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
			return 0.0

		area_i = Utils.intersection(a, b)
		area_u = Utils.union(a, b, area_i)
		return float(area_i) / float(area_u + 1e-6)


	@classmethod
	def get_new_img_size(cls,width, height, img_min_side=600):
		""" Compute new image size """
		if width <= height:
			f = float(img_min_side) / width
			resized_height = int(f * height)
			resized_width = img_min_side
		else:
			f = float(img_min_side) / height
			resized_width = int(f * width)
			resized_height = img_min_side

		return resized_width, resized_height


