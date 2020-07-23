import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import mean_shift as ms
import mean_shift_tools as tools
import mean_shift_flood_fill as fill
import sys
sys.setrecursionlimit(10000)

hs = 32
hr = 40
IMAGE_SIZE_X = 225
IMAGE_SIZE_Y = 225
factor = 0

resize_factor = math.pow(2,factor)
SIZE_X = round(IMAGE_SIZE_X/resize_factor)
SIZE_Y = round(IMAGE_SIZE_Y/resize_factor)

IMAGE_DIR = '/image.png'
#IMAGE_DIR = '/MeanShift_py-master/sample_images/mean_shift_image.jpg'
image = Image.open(IMAGE_DIR)
image_raw = np.array(image)
image = np.array(image.resize((SIZE_X,SIZE_Y)))
image_padded = tools.padding(image, hs)

'''discontinuity preserving filtering'''
#mean_shifter = ms.MeanShift(kernel='gaussian_kernel')
mean_shifter = ms.MeanShift(kernel='uniform_kernel')
image_filtered = mean_shifter.cluster(image_padded, bandwidth=[hs,hr])

plt.imshow(image_filtered) 
#plt.imshow(image) 

'''fllod filling'''
# FloodFill = fill.FloodFill()
# image_final = FloodFill.flood_fill(image_filtered)
# plt.imshow(image_final)



    














