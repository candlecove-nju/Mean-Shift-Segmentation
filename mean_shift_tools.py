import math
import numpy as np
#from scipy import stats

def euclidean_dist(pointA, pointB):
    total = float(0)
    for dimension in range(0, len(pointA)):
        total += (pointA[dimension] - pointB[dimension])**2
    return math.sqrt(total)

def gaussian_kernel(distance, bandwidth):
    euclidean_distance = np.sqrt(((distance)**2).sum())
    val = (1/(np.pow(bandwidth,3)*math.sqrt(2*math.pi))) * np.exp(-0.5*((euclidean_distance / bandwidth))**2)*euclidean_distance
    return val

def uniform_kernel(distance, bandwidth):
    val=0
    temp = np.zeros([1,3])
    if (euclidean_dist(distance, temp[0]) <= bandwidth):
        val=1
    return val

def padding(image, width):
    H, W = image.shape[0], image.shape[1] 
    image_padded = np.zeros([H+2*width, W+2*width, 3])
    image_padded[width:H+width, width:W+width, :] = image
    return image_padded



        





