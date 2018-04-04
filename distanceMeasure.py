# import the necessary packages
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory of images")
args = vars(ap.parse_args())
 
# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}

# loop over the image paths
for imagePath in glob.glob(args["dataset"] + "/*.png"):
	# extract the image filename (assumed to be unique) and
	# load the image, updating the images dictionary
	filename = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
	# extract a 3D RGB color histogram from the image,
	# using 8 bins per channel, normalize, and update
	# the index
	hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
	cv2.normalize(hist, hist)
	hist = hist.flatten()
	index[filename] = hist

# METHOD #2: UTILIZING SCIPY
# initialize the scipy methods to compaute distances
SCIPY_METHODS = (
	("Euclidean", dist.euclidean),
	("Manhattan", dist.cityblock),
	("Chebysev", dist.chebyshev))

# loop over the comparison methods
returnDistance = 0
for (methodName, method) in SCIPY_METHODS:
	# initialize the dictionary dictionary
	results = {}

	# loop over the index
	for (k, hist) in index.items():
		# compute the distance between the two histograms
		# using the method and update the results dictionary
		d = method(index["comparison\\imageInput.png"], hist)
		results[k] = d

	# sort the results
	results = sorted([(v, k) for (k, v) in results.items()])

	# loop over the results
	for (i, (v, k)) in enumerate(results):
		while i < len(results) - 1:
			i += 1
			returnDistance = returnDistance + float(results[i][0])
	# smaller distance is better match
print(returnDistance)