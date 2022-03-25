import numpy as np
import skimage as sk


def get_break_vals(alg_obj, return_sat=True):
    
    pcap, sat = alg_obj.get_intrusion_data()

    for i, p in enumerate(pcap):
        if alg_obj.is_percolating(p):
            p_break = p
            sat_break = sat[i]
            break

    return p_break, sat_break

def get_approx_sat_pcap(sat, perc_obj):
    
    pcaps, sats = perc_obj.get_intrusion_data()
    
    sat_dist = np.abs(np.array(sats) - sat)
    nearest_sat_idx = np.argmin(sat_dist)
    
    return pcaps[nearest_sat_idx]

def get_approx_pcap_sat(pcap, perc_obj):
    
    pcaps, sats = perc_obj.get_intrusion_data()
    
    pcap_dist = np.abs(np.array(pcaps) - pcap)
    nearest_pcap_idx = np.argmin(pcap_dist)
    
    return sats[nearest_pcap_idx]



def get_sat_pore_throat(sat, perc_obj, pn, geo, invasion_type=None):
    
    pcap = get_approx_sat_pcap(sat, perc_obj)
    
    if invasion_type == 'invasion':
        filled_pore_mask = np.bool_(perc_obj.results(sat)['pore.occupancy'])
        filled_throat_mask = np.bool_(perc_obj.results(sat)['throat.occupancy'])
        
    else:
        pcap = get_approx_sat_pcap(sat, perc_obj)
        filled_pore_mask = np.bool_(perc_obj.results(pcap)['pore.occupancy'])
        filled_throat_mask = np.bool_(perc_obj.results(pcap)['throat.occupancy'])
        
#     sat_pore_mask = pn['pore._id'][filled_pore_mask]
#     sat_throat_mask = pn['throat._id'][filled_throat_mask]
    sat_pore_mask = filled_pore_mask
    sat_throat_mask = filled_throat_mask
    
    return sat_pore_mask, sat_throat_mask


def create_slc(pn, geo, n_slice=50):
    
    # pores
    z = pn['pore.coords'][:, 0]
    z_min, z_max = z.min(), z.max()
    
    bins = np.linspace(z_min, z_max, n_slice)
    pore_slc = np.digitize(z, bins=bins)
    
    # throats
    z_t = geo['throat.endpoints.head'][:, 0]
#     z_t_min, z_t_max = z_t.min(), z_t.max()
    
#     bins = np.linspace(z_t_min, z_t_max, n_slices)
    t_slc = np.digitize(z_t, bins=bins)
    
    return pore_slc, t_slc

def get_sphere_spatial_h(centre, radius):
    
    r_pos, c_pos = centre
    grid_r, grid_c = sk.draw.disk(center=(r_pos, c_pos), radius=radius)
    
    p_dist = np.sqrt((grid_r-r_pos)**2 + (grid_c-c_pos)**2)
    thickness = 2*np.sqrt(radius**2 - p_dist**2)
    
    return thickness, grid_r, grid_c

def get_cylinder_proj(head, tail, throat_diams):
    
    two_d_proj_length = np.linalg.norm(head[:, 1:]-tail[:, 1:], axis=1)

    head_r = head[:, 1]
    head_c = head[:, 2]

    tail_r = tail[:, 1]
    tail_c = tail[:, 2]
    
    # find unit vector perp to throat segmentt
    # https://stackoverflow.com/questions/133897/how-do-you-find-a-point-at-a-given-perpendicular-distance-from-a-line
    line_vector = np.array([tail_r-head_r, tail_c-head_c]).T
    unit_vect = line_vector /  two_d_proj_length.reshape(-1, 1) 
    rot_mat = np.array([[0, -1], [1, 0]])
    unit_vect = np.matmul(unit_vect,rot_mat)  # rotate to perpendicular vector

    top_left_x = head_r + unit_vect[:, 0]*(throat_diams/2)
    top_left_y = head_c + unit_vect[:, 1]*(throat_diams/2)

    # add negative of unit vector from the tail
    bot_right_x = tail_r - unit_vect[:, 0]*(throat_diams/2)
    bot_right_y = tail_c - unit_vect[:, 1]*(throat_diams/2)
    
    return (top_left_x, top_left_y), (bot_right_x, bot_right_y)

def get_cylinder_spatial_h(top_left_loc, bot_right_loc, diam):
    
    grid_r, grid_c = sk.draw.rectangle(start=top_left_loc,
                                      end=bot_right_loc,)
    
#     r_pos, c_pos = centre
#     radius = diam/2.0
    
#     p_dist = np.sqrt((grid_r-r_pos)**2 + (grid_c-c_pos)**2)
#     thickness = 2*np.sqrt(radius**2 - p_dist**2)
    thck = diam/2  # compensate for too much thickness
    
    return thck, grid_r, grid_c



def sample_cap(OP, sat_step, percolation_type=None ):
    
    # get saturation steps
    data = OP.get_intrusion_data()
    
    # check the type of percolation whether invasion or ordinary
    if percolation_type == 'invasion':
        sat_pts = [data.S_tot[0]]; pc_points = [data.Pcap[0]]  # initialize with first points
        
        
        # sample the saturation data
        for i, pc in enumerate(data.Pcap):
            if data.S_tot[i] - sat_pts[-1] >= sat_step:
                sat_pts.append(data.S_tot[i])
                pc_points.append(pc)
                
        # add last saturation point        
        sat_pts.append(data.S_tot[-1])
        pc_points.append(data.Pcap[-1])
    else:
        sat_pts = [data.Snwp[0]]; pc_points = [data.Pcap[0]]  # initialize with first points
        
        # sample the saturation data
        for i, pc in enumerate(data.Pcap):
            if data.Snwp[i] - sat_pts[-1] >= sat_step:
                sat_pts.append(data.Snwp[i])
                pc_points.append(pc)
                
        # add last saturation point
        sat_pts.append(data.Snwp[-1])
        pc_points.append(data.Pcap[-1])

    return sat_pts, pc_points


def get_thck_map(ps, ts, pn, geo,
                 px_val, im_shape, map_throat=True, match_2d=True):
    
    
    px_coord = (pn['pore.coords'] / px_val)[ps]
    diams = (geo['pore.diameter'] / px_val)[ps]
    # fill pore water ========================
    
    
    # initialize holding image
    nrow, ncol = im_shape
    im_water_proj = np.zeros((nrow, ncol), dtype=np.float32)


    for i, loc in enumerate(px_coord):

        c_pos = int(px_coord[i, 2])
        r_pos = int(px_coord[i, 1])
        radius = diams[i]/2

        thck, grid_r, grid_c = get_sphere_spatial_h(centre=(r_pos, c_pos), 
                                                    radius=radius)

        valid_dims = (grid_r < nrow) & (grid_c < ncol)

        grid_r = grid_r[valid_dims]
        grid_c = grid_c[valid_dims]
        thck = thck[valid_dims]

        im_water_proj[grid_r, grid_c] += thck

    if map_throat is True:    
    # fill throat water =======================
        head = (geo['throat.endpoints.head'] / px_val)[ts]
        tail = (geo['throat.endpoints.tail'] / px_val)[ts]
        throat_diams = (geo['throat.diameter'] / px_val)[ts]
        
        top_left, bot_right = get_cylinder_proj(head, tail, throat_diams)
        top_left_x = top_left[0]
        top_left_y = top_left[1]
        bot_right_x = bot_right[0]
        bot_right_y = bot_right[1]
        
        for i in range(len(head)):
            
            top_left = int(top_left_x[i]), int(top_left_y[i])
            bot_right = int(bot_right_x[i]), int(bot_right_y[i])
            diam = throat_diams[i]
            
            
            thck, grid_r, grid_c =  get_cylinder_spatial_h(top_left, bot_right, diam)
            
            valid_dims = (grid_r < nrow) & (grid_c < ncol)

            grid_r = grid_r[valid_dims]
            grid_c = grid_c[valid_dims]
            

            im_water_proj[grid_r, grid_c] += thck       
    #=========================================
    
                                                                        
    # transform to match radiograph image
    if match_2d is True:
        im_water_proj = sk.transform.rotate(im_water_proj, angle=180)
    
    return im_water_proj

  
