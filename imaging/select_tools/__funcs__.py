import numpy as np

def select_phase(im, phase_val):
    
    if isinstance(phase_val, int):
    
        im_ = np.where(im==phase_val, 1, 0)
        
    # elif isinstance(phase_val, tuple):
        
    #     n_phases = len(phase_val)
    #     for i in range(n_phases):
    #         im = np.where(im==phase_val[i], 1, im)
            
    #     im_= np.where(im==1, im, 0)
        
    return im_

def cut_equal_yz(im_1, im_2):
    
    """set 3D array to equal_dimensions, changes are slight and will
    typically be oly in the y and z plane (x is the number of slices which should be equal)"""
    
    shape_1 = im_1.shape
    shape_2 = im_2.shape
    
    min_y = np.min([shape_1[1], shape_2[1]])
    min_z = np.min([shape_1[2], shape_2[2]]) 
    
    im_1 = im_1[:, :min_y, :min_z].copy()
    im_2 = im_2[:, :min_y, :min_z].copy()
    
    return im_1, im_2

def cut_equal_xy_2D(im_1, im_2):
    
    """set 3D array to equal_dimensions, changes are slight and will
    typically be oly in the y and z plane (x is the number of slices which should be equal)"""
    
    shape_1 = im_1.shape
    shape_2 = im_2.shape
    
    min_x = np.min([shape_1[0], shape_2[0]])
    min_y = np.min([shape_1[1], shape_2[1]]) 
    
    im_1 = im_1[ :min_x, :min_y].copy()
    im_2 = im_2[ :min_x, :min_y].copy()
    
    return im_1, im_2

def rect_crop(im, rect_spec):

    """following imageJ style rectangle specification"""

    left, top, width, height = rect_spec

    im_crop = im[:, top:top+height, left:left+width].copy()

    return im_crop

def rect_crop_2D(im, rect_spec):

    """following imageJ style rectangle specification"""

    left, top, width, height = rect_spec

    im_crop = im[top:top+height, left:left+width].copy()

    return im_crop

def get_imagej_rect_centre(rect_spec):
    
    x = rect_spec[0] + rect_spec[2] // 2
    y = rect_spec[1] + rect_spec[3] // 2
    
    return x, y

def get_imagej_rect_props(rect_spec):

    x = rect_spec[0] + rect_spec[2] / 2
    y = rect_spec[1] + rect_spec[3] / 2

    # centre = (x, y)

    height = rect_spec[3]
    width = rect_spec[2]

    bottom_left = rect_spec[0],  rect_spec[1]+rect_spec[3]
    top_left = rect_spec[0], rect_spec[1]

    props = {'centre': (x,y), 'height':height, 'width':width, 'top-left':top_left, 'bottom-left':bottom_left}

    return props