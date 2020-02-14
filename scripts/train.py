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

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections


## MODULES
from sidelobe_finder import __version__, __date__
from sidelobe_finder import logger
from sidelobe_finder.config import Config
from sidelobe_finder.data_provider import DataProvider
from sidelobe_finder.nn_trainer import NNTrainer
from sidelobe_finder.usernet import NetworkModel


#### GET SCRIPT ARGS ####
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-data', '--data', dest='datalist', required=True, type=str,action='store',help='List of files with data images and bounding box info')
	parser.add_argument('-config', '--config', dest='config', required=False, type=str,default='', action='store',help='Configuration file')	
	parser.add_argument('-loglevel','--loglevel', dest='loglevel', required=False, type=str, default='INFO', help='Logging level (default=INFO)') 
	

	args = parser.parse_args()	

	return args



##############
##   MAIN   ##
##############
def main():
	"""Main function"""

	
	#===========================
	#==   PARSE ARGS
	#===========================
	logger.info("Get script args ...")
	try:
		args= get_args()
	except Exception as ex:
		logger.error("Failed to get and parse options (err=%s)",str(ex))
		return 1

	# - Input filelist
	datalist= args.datalist
	config_file= args.config
	#nnarcfile= args.nnarcfile

	log_level_str= args.loglevel.upper()

	log_level = getattr(logging, log_level_str, None)
	if not isinstance(log_level, int):
		logger.error('Invalid log level given: ' + log_level_str)
		return 1
	logger.setLevel(log_level)

	#===========================
	#==   PARSE CONFIG
	#===========================
	config= Config()

	if config_file:
		logger.info("Parse config file ...")
		status= config.parse(config_file)	
		if status<0:
			logger.error("Failed to parse and validate config file " + config_file + "!")
			return 1

	#===========================
	#==   READ DATA
	#===========================
	dp= DataProvider(datalist,config)

	logger.info("Reading input data ...")
	status= dp.read_data()	
	if status<0:
		logger.error("Failed to read input data listed in file %s!" % (datalist))
		return 1


	#===========================
	#==   BUILD NETWORK
	#===========================
	if config.network == 'vgg':
		from sidelobe_finder import vgg as nn
	elif config.network == 'resnet50':
		from sidelobe_finder import resnet as nn
	elif config.network == 'user':
		nn= NetworkModel(dp,config)
		if nn.build_network()<0:
			logger.error("Failed to build user network model!")
			return -1
	else:
		logger.error("Invalid network model given!")
		return 1

	#===========================
	#==   TRAIN NN
	#===========================
	logger.info("Running R-CNN training ...")
	
	t= NNTrainer(dp,nn,config)
	
	status= t.run_train()
	if status<0:
		logger.error("Failed to run train!")
		return 1

	# ...
	# ...

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

