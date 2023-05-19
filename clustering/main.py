#!/usr/bin/env python3

from KMean_Clustering import KMean
import numpy as np
import cv2

def main():
    image_path = r"../data/Tho2.jpeg"
    image = cv2.imread(image_path)
    
    # Convert the image to RGB format if it's in BGR format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_array = image.reshape(-1, 3)
    
    kmeans = KMean(3)
    
    kmeans.fit(image_array)
    
    labels = kmeans.predict(image_array)
    
    # Plot the image with the clusters colored differently.
    clustered_image = np.zeros_like(image_array)
    for i in range(kmeans.n_clusters):
        mask = labels == i
        color = kmeans.centroids[i]
        clustered_image[mask] = color
    
    clustered_image = clustered_image.reshape(image.shape)
    
    clustered_image = clustered_image.astype(np.uint8)
    
    cv2.imshow("Clustered Image", clustered_image)
    cv2.waitKey(0)
if __name__ == '__main__':
	main()
