#!/bin/env python
# Code inspired by from https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/ and some other opencv tutorials

import numpy as np
import argparse
import cv2
import os
import pdb

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
ap.add_argument("-z", "--zoom", help = "magnification")
ap.add_argument("-d", "--dir", help = "process all *jpg and *png files in a directory")
args = vars(ap.parse_args())

zoomToAreaMapping = {
	"100": 1000,
	"50": 100,
	"20": 20,
	"10": 20,
	"5": 5
}

zoomToColourMapping = {  # remember these triplet values are BGR (reversed RGB)
	"100": ([197, 121, 171], [212, 138, 182]), # based on flake2_x0.jpg
	"50": ([202, 135, 176], [213, 142, 182]), # based on flake_x50.jpg
	# "50": ([197, 121, 171], [213, 142, 182]), # based on flake2_x50.jpg
	"20": ([219, 83, 91], [236, 96, 102]), #based on flake_x20.jpg
	# "20": ([219, 80, 82], [228, 88, 92]),	# based on flake2_x20.jpg
	"10": ([219, 93, 88], [235, 100, 102]),	# based on flake_x10.jpg
	# "10": ([219, 83, 83], [228, 94, 94]),	# based on flake2_x10.jpg
	"5": ([225, 90, 96], [229, 93, 102])	# based on flake2_x5.jpg
}



def process_image(image_file):

	minArea = None
	# try detecting the proper zoom from the file name if no zoom is set
	if args["zoom"] is None:
		for index,zoom in zoomToAreaMapping.items():
			# pdb.set_trace()
			if "x"+index+"." in image_file:
				minArea = zoom
				lower, upper = zoomToColourMapping.get(index, None)
				# pdb.set_trace()
				break
	else:
		# set the minimal area to be considered. The second parameter of the 'get' method is the default value.
		minArea = zoomToAreaMapping.get(args["zoom"], None)
		# define the colour boundaries
		lower, upper = zoomToColourMapping.get(args["zoom"], None)
	
	if minArea is None:
		return
	
	# load the image
	image = cv2.imread(image_file)

	

	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")

	# find the colors within the specified boundaries and apply the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)

	# convert image to grayscale
	gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

	# stabilize the image to only 0 and 255
	thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]

	# find contours, limit them per their area and draw them
	_, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours2 = [c for c in contours if cv2.contourArea(c) > minArea]
	cv2.drawContours(output, contours2, -1, (0,255,0), 3)

	# show the images
	# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('image', 1920, 1080)
	# cv2.imshow("image", output)
	# cv2.waitKey(0)

	cv2.imwrite(image_file + "_processed.png", output)

if args["image"] is not None:
	process_image(args["image"])
else:
	if args["dir"] is None: args["dir"] = "./"
	directory = os.fsencode(args["dir"])
	for filename in os.listdir(args["dir"]):
		if (filename.endswith(".jpg") or filename.endswith(".png")) and "_processed" not in filename:
			process_image(args["dir"] + filename)
		else:
			continue

