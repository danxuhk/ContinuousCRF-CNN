import numpy as np
import random
import scipy.ndimage as ndimage
import cv2


class DataTransformer:

    """
    DataTransformer is a simple class for preprocessing and deprocessing
    images for caffe.
    """

    def __init__(self, mean_values=[104.008, 116.669, 122.675], crop_size=400):
        self.mean_values = np.array(mean_values, dtype=np.float32)
        self.crop_size = crop_size;
    
    
    def scale_img(self, im, new_h, new_w, interpolation_, is_depth, scale=0):
        
        if not is_depth:
            # scale input image
            return cv2.resize(im, (new_h, new_w), interpolation=interpolation_);
        else:
            return cv2.resize(im, (new_h, new_w), interpolation=interpolation_) / float(scale); 
                  
        
    def random_crop_img(self, scaled_img, rand_y, rand_x):        
            
        # return cropped image (3-channel or 1-channel image)
        if len(scaled_img.shape) == 3:
            return scaled_img[rand_y:rand_y+self.crop_size, rand_x:rand_x+self.crop_size, :];
        else:
            return scaled_img[rand_y:rand_y+self.crop_size, rand_x:rand_x+self.crop_size];
            
             
    def preprocess(self, im, seg_label, depth_label, surface_label, contour_label, scale):
        """
        preprocess() emulate the pre-processing occurring in the vgg16 caffe
        prototxt.
        """
        ##process input image
        assert im.dtype == np.uint8;
        im = np.float32(im)
        im = im[:, :, ::-1]  # change to BGR
        # subtract mean values
        # im[:, :, 0] -= self.mean_values[0];
        # im[:, :, 1] -= self.mean_values[1];
        # im[:, :, 2] -= self.mean_values[2];
        im -= self.mean_values;
        
        ###### rescale images ########
        
        # generate a new scale (new_h, new_w)
        new_h = int(im.shape[0] * float(scale));
        new_w = int(im.shape[1] * float(scale));
        interpolation_ = cv2.INTER_LINEAR;
        im = self.scale_img(im, new_h, new_w, interpolation_, False);
        
        ##process segmentation label map
        assert seg_label.dtype == np.uint8;
        interpolation_ = cv2.INTER_NEAREST;
        seg_label = self.scale_img(seg_label, new_h, new_w, interpolation_, False);
        
        ##process depth map
        assert depth_label.dtype == np.uint16;
        depth_label = depth_label / float(1000);
        interpolation_ = cv2.INTER_LINEAR;
        depth_label = self.scale_img(depth_label, new_h, new_w, interpolation_, True, scale);
        
        ##process surface normal map
        assert surface_label.dtype == np.uint8;
        interpolation_ = cv2.INTER_LINEAR;
        surface_label = self.scale_img(surface_label, new_h, new_w, interpolation_, False);
        
        #process contour map
        assert contour_label.dtype == np.uint8;
        interpolation_ = cv2.INTER_NEAREST;
        contour_label = self.scale_img(contour_label, new_h, new_w, interpolation_, False);
        
        ##### crop rescaled images #####
        
        # generate a random starting point for all images
        rand_y = random.randint(0, im.shape[0] - self.crop_size);
        rand_x = random.randint(0, im.shape[1] - self.crop_size);
        # rgb
        im = self.random_crop_img(im, rand_y, rand_x);
        # seg
        seg_label = self.random_crop_img(seg_label, rand_y, rand_x);
        #depth
        depth_label = self.random_crop_img(depth_label, rand_y, rand_x);
        # surface normal
        surface_label = self.random_crop_img(surface_label, rand_y, rand_x);
        # contour
        contour_label = self.random_crop_img(contour_label, rand_y, rand_x);
               
        im = im.transpose((2, 0, 1));
        surface_label = surface_label.transpose((2, 0, 1));
        #seg_lable = seg_label[np.newaxis, ...];
        seg_label = seg_label.reshape(1, self.crop_size, self.crop_size);
        depth_label = depth_label.reshape(1, self.crop_size, self.crop_size);
        contour_label = contour_label.reshape(1, self.crop_size, self.crop_size);

        return im, seg_label, depth_label, surface_label, contour_label