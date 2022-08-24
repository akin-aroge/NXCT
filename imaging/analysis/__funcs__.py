import imaging as ig
import numpy as np
import skimage as sk


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


def get_connectivity(im, phase_val, slack=3, return_all=None):

    """returns im_conn_label, im_span_label, comp_span_dict, connectivity_key"""
    
    im_seg = ig.select_tools.select_phase(im, phase_val=phase_val)

    full_span = im_seg.shape[0]

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
            conn_label = 1
        elif loc_z[0] <= slack:
            conn_label = 2
        elif loc_z[-1] >= full_span-slack:
            conn_label = 3
        else:
            conn_label = 4
        
        im_conn_label[loc_z, loc_x, loc_y] = conn_label
    
    
    return im_conn_label, im_span_label, comp_span_dict, connectivity_key