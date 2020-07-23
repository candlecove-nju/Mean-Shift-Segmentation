import numpy as np
import mean_shift_tools as tools

MIN_DISTANCE = 0.1
ITERATIONS_MAX = 5

class MeanShift(object):
    
    def __init__(self, kernel=tools.uniform_kernel):
        if kernel == 'uniform_kernel':
            kernel = tools.uniform_kernel
        if kernel == 'gaussian_kernel':
            kernel = tools.gaussian_kernel
        self.kernel = kernel
        
    def cluster(self, feature_space, bandwidth):
        if (len(bandwidth) != 2):
            raise Exception("Two inputs!")
        hs = bandwidth[0];
        hr = bandwidth[1];
        H, W = feature_space.shape[0], feature_space.shape[1]
        shifted_points_pos = np.zeros([H,W,2])
        one_step_pos = np.zeros([H,W,2])
        for i in range(hs,H-hs):
            for j in range(hs,W-hs):   
                print("{} {}".format(i,j)) 
                center = np.array([i,j])
                spatial_block = feature_space[i-hs:i+hs+1, j-hs:j+hs+1, :]
                one_step_pos[i,j,:] = self.shift_point(center, spatial_block, hr) 
        for i in range(hs,H-hs):
            for j in range(hs,W-hs):   
                print('Mean Shift',i,j)       
                center_new = one_step_pos[i,j,:]
                #print(center_new)
                dist = tools.euclidean_dist(center, center_new)
                num = 0
                while dist > MIN_DISTANCE:
                    num+=1
                    center = center_new                
                    center_new = one_step_pos[int(center[0]),int(center[1])]           
                    dist = tools.euclidean_dist(center, center_new)
                    if (num > ITERATIONS_MAX):
                        break
                shifted_points_pos[i,j] = np.round(center)             
        image_filtered = np.zeros(feature_space.shape)
        for i in range(H):
            for j in range(W):
                image_filtered[i,j,:] = feature_space[int(shifted_points_pos[i,j,0]),int(shifted_points_pos[i,j,1]),:]
        image_filtered = image_filtered[hs:H-hs,hs:W-hs,:].astype(np.uint8)
        return image_filtered
    
    def shift_point(self, center, spatial_block, hr):
        w = spatial_block.shape[0];
        positions = np.zeros([w, w, 2])
        center_position = np.array([(w-1)/2, (w-1)/2]).astype(int)
        positions[int(center_position[0]), int(center_position[1]), :] = center
        for i in range(w):
            for j in range(w):
                if (i != (w-1/2))and(j != (w-1/2)):
                    positions[i,j] = center+np.array([i,j])-center_position 
        positions = positions.reshape((w**2, 1, 2))  
        vectors = (spatial_block-spatial_block[center_position[0],center_position[1]]).reshape((w**2, 1, 3))
        points_weight = np.zeros([w**2, 1])
        for i in range(w**2):
            points_weight[i,0] = self.kernel(vectors[i,0,:], hr)
        shifted_point_x = sum(np.multiply(points_weight, positions[:,:,0]))
        shifted_point_y = sum(np.multiply(points_weight, positions[:,:,1]))
        shifted_point = np.array([shifted_point_x[0], shifted_point_y[0]]/sum(points_weight))
        return shifted_point


        

