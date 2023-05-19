#!/usr/bin/env python3

from KMean_Clustering import KMean
import numpy as np
import cv2

def main():
	image_path = r"../data/thobaymau.jpg"
	image = cv2.imread(image_path)
	image_array = np.array(image)
	image_array = image_array.reshape(-1, image_array.shape[-1])

	KMean(5)

if __name__ == '__main__':
	main()

