import numpy as np
import mean_shift_tools as tools

THRESHOLD = 100

class FloodFill(object):
    
    def flood_fill(self, image):
        H, W = image.shape[0], image.shape[1]
        mask = np.zeros([H,W])
        index = 1
        for i in range(1,H-1):
            for j in range(1,W-1):
                if (mask[i,j] == 0):
                    center_color = image[i,j]
                    mask, image = self.region_grow(mask, image, index, [i,j], center_color)
                    index += 1
        return image
                    
    def region_grow(self, mask, image, index, center, center_color):
        H, W = mask.shape
        if ((center[0]==0)or(center[0]==H-1)or(center[1]==0)or(center[1]==W-1)):
            return mask, image
        mask[center[0],center[1]] = index
        image[center[0],center[1]] = center_color  
        for i in range(center[0]-1,center[0]+2):
            for j in range(center[1]-1,center[1]+2): 
                if (mask[i,j] !=0):
                    continue
                else:
                    dist = tools.euclidean_dist(center_color, image[i,j])
                    if (dist < THRESHOLD):
                        mask, image = self.region_grow(mask, image, index, [i,j], center_color)               
        return mask, image
    