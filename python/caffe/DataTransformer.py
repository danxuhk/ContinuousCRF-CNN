import numpy as np
import random
import scipy.ndimage as ndimage


class DataTransformer:

    """
    DataTransformer is a simple class for preprocessing and deprocessing
    images for caffe.
    """

    def __init__(self, mean=[128, 128, 128]):
        self.mean = np.array(mean, dtype=np.float32)
        self.scale = 1.0
    
    def scale_img(self, im, is_depth):
        if not is_depth:
            # scale input image
            return cv2.resize(im, (im.shape[0] * float(self.scale), im.shape[1] * float(self.scale), im.shape[2]).astype(np.uint8));
        else:
            return (cv2.resize(im, (im.shape[0] * float(self.scale), im.shape[1] * float(self.scale), im.shape[2]).astype(np.uint8))) / float(self.scale);       
        
    def random_crop_img(self, im):        
        # crop image
        rand_y = random.randint(0, scaled_img.shape[0] - self.crop_size);
        rand_x = random.randint(0, scaled_img.shape[1] - self.crop_size);
        # return cropped image
        return scaled_img[rand_y:rand_y+self.crop_size-1, rand_x:rand_x+self.crop_size-1, :];
             
    def preprocess(self, im, seg_label, depth_label, surface_label, contour_label):
        """
        preprocess() emulate the pre-processing occurring in the vgg16 caffe
        prototxt.
        """
        ##process input image
        assert im.dtype == np.uint8;
        im = np.float32(im)
        im = im[:, :, ::-1]  # change to BGR
        im -= self.mean
        # resize and random crop
        im = scale_img(im, False);
        im = random_crop_img(im);
        
        ##process segmentation label map
        assert seg_label.dtype == np.uint8;
        seg_label = scale_img(seg_label, False);
        seg_label = random_crop_img(seg_label);
        
        ##process depth map
        assert depth_label.dtype == np.uint16;
        depth_label = depth_label / float(1000);
        depth_label = scale_img(depth_label, True);
        depth_label = random_crop_img(depth_label);
        
        ##process surface normal map
        assert surface_label.dtype == np.uint16;
        surface_label = scale_img(surface_label, False);
        surface_label = random_crop_img(surface_label);
        
        #process contour map
        assert contour_label.dtype == np.uint8;
        contour_label = scale_img(contour_label, False);
        contour_label = random_crop_img(contour_label);
               
        im = im.transpose((2, 0, 1));
        seg_label = seg_label.reshape(1, self.crop_size, self.crop_size,);
        depth_label = depth_label.reshape(1, self.crop_size, self.crop_size,);
        surface_label = surface_label.reshape(1, self.crop_size, self.crop_size,);
        contour_label = contour_label.reshape(1, self.crop_size, self.crop_size,);

        return im, seg_label, depth_label, surface_label, contour_label