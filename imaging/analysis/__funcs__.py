import imaging as ig
import numpy as np
import skimage as sk


def ccm_thickness(im_cl_seg, px_val):


    def cl_seg_thk(im_cl_seg, px_val):
        """calculates the CCM thickness from the segmented
        anode and cathode catalyst layer.
        """

        top = np.argmax(im_cl_seg, axis=0)
        n_z = im_cl_seg.shape[0]
        bot = n_z - 1 - np.argmax(im_cl_seg[::-1], axis=0)

        t = bot - top
        filter_ = t < (n_z-1)
        t_vals = t[filter_]

        t_map = t.copy()
        t_map[~filter_] = 0

        t_map = t_map * px_val
        t_vals = t_vals * px_val
        return t_vals, t_map
    
    t_vals, t_map = cl_seg_thk(im_cl_seg=im_cl_seg, px_val=px_val)

    t_vals = np.expand_dims(t_vals, axis=1)
    fig, ax, mus, stds, ws = ig.tools.gauss_fit1D(data=t_vals, n_comp=2, comps=['a', 'b'] )
    thk_mean = np.max(mus)
    idx_max_mean = np.argmax(mus)
    thk_std = stds[idx_max_mean]

    return thk_mean, thk_std, t_map

def calc_mem_thk(acl_seg, ccl_seg, px_val):
    
    n_z = acl_seg.shape[0]
    
    a_bot = n_z - 1 - np.argmax(acl_seg[::-1], axis=0)
    a_bot[a_bot==(n_z-1)] = 0  # remove regions where there is no seg comp 
    
    c_top = np.argmax(ccl_seg, axis=0)
    c_top[c_top==(n_z-1)] = 0  # this not necessary really
    
    t = c_top - a_bot
    
    # get where CCL and ACL seg both appear throughplane
    sample_filter = np.bool_(c_top) & np.bool_(a_bot)
    
    t_vals = t[sample_filter]
    t_vals = t_vals*px_val
    
    t_map = t.copy()
    t_map[~sample_filter] = 0
    t_map = t_map*px_val
    
    return t_vals, t_map


def get_connectivity(im, phase_val, slack=3, full_span=None, return_all=None):

    """returns im_conn_label, im_span_label, comp_span_dict, connectivity_key"""
    
    im_seg = ig.select_tools.select_phase(im, phase_val=phase_val)

    img_span = im_seg.shape[0]

    if full_span is None:
        full_span = img_span

    

    if full_span < np.iinfo('uint8').max:
        data_type = 'uint8'
    else:
        data_type = 'uint16'



    connectivity_key = {1:'full', 2:'top', 3:'bottom', 4:'not-conn.' }  # air is 0

    water_comp_labels, comp_count = sk.morphology.label(im_seg, return_num=True)


    comp_labels = np.arange(1, comp_count+1)  # make sure zero label is handled when you do this

    comp_span_dict = {}
    # create image where each pixel is labeled with the span of its compnent
    im_span_label = np.zeros_like(im_seg, dtype=data_type)  # zero label is handled
    im_conn_label = np.zeros_like(im_seg, dtype=data_type)
    
    for comp_label in comp_labels:
        loc_z, loc_x, loc_y = np.where(water_comp_labels == comp_label)

        #comp_len = np.max(loc_z) - np.min(loc_z)
        comp_tp_span = loc_z[-1] - loc_z[0] + 1   # since the labelling is ordered in the z direction, +1 for 1 slice

        comp_span_dict[comp_label] = comp_tp_span

        im_span_label[loc_z, loc_x, loc_y] = comp_tp_span
       
        
        # connectivity category:
        
        
        # order matters here
        if (loc_z[-1] - loc_z[0]) >= full_span - slack:
            conn_label = 1  #full
        # for top connected (2) the if img_span < full span
        # then there cant be topconnected
        elif (img_span >= full_span) and (loc_z[0] <= slack):
            conn_label = 2  #top
        elif loc_z[-1] >= img_span-slack:
            conn_label = 3  #bottom
        else:
            conn_label = 4  #not
        
        im_conn_label[loc_z, loc_x, loc_y] = conn_label
    
    
    return im_conn_label, im_span_label, comp_span_dict, connectivity_key



def get_phase_vol(im, phase_val, ax):

    # if px_val is None:
    #     px_val = 1.54 #typical for the experiment

    dim = im.shape
    
    if len(dim) == 2:
        im = im.reshape(-1, dim[0], dim[1])
        dim = im.shape
        
    im_ = np.where(im == phase_val, 1, 0)
    
    if ax == 'x':
        result = np.sum(im_, axis=0).sum(axis=0) 
        
    elif ax == 'y':
        result = np.sum(im_, axis=0).sum(axis=1) 
    
    elif ax == 'z':
        result = np.sum(im_, axis=1).sum(axis=1) 

    elif ax is None:
        result = np.mean(im_)
        
    #vol = result * (px_val**3)
        
    return result
