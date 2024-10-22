import numpy as np
import skimage as sk
import openpnm as op
import porespy as ps
import matplotlib.pyplot as plt
# import openpnm as op

def compute_psd(im, pixel_size):

    # setup workspace
        
    ws = op.Workspace()
    ws.settings["loglevel"] = 50
    ws.clear()
    proj = ws.new_project(name='proj')

    # extract network
    net = ps.networks.snow(im=im, voxel_size=pixel_size)
    op.io.PoreSpy.import_data(net, project=proj)

    # trim pores
    pore_health = proj.network.check_network_health()
    op.topotools.trim(network=proj.network, pores=pore_health['trim_pores'])

    fig, ax = plt.subplots()
    ax.hist(net['pore.diameter'],  bins=40, alpha=0.7,edgecolor='k', density=True, label='channel')
    ax.set_xlabel('pore diameter ($\mu m$)')
    ax.set_ylabel('proportion')
    
    return net['pore.diameter']

def run_percolation(project, inlet='left', outlet='right', inv_phase='water', phase_theta=110, invasion=False, n_pts=100):

    if isinstance(inlet, str):
        pn = project.network
        inlets = pn.pores(inlet)
        outlets = pn.pores(outlet)
    else:
        inlets = inlet
        outlets = outlet

    if invasion:
        net = project.network
        OP = op.algorithms.InvasionPercolation(network=net)
    else:
        OP = op.algorithms.OrdinaryPercolation(project=project)

    phase = project.phases()[inv_phase]
    phase['pore.contact_angle'] = phase_theta
    phase.regenerate_models(deep=True)

    if invasion:
        OP.setup(phase=phase, pore_volume='pore.volume', throat_volume='throat.volume')
    else:
        OP.setup(phase=phase, pore_volume='pore.volume', throat_volume='throat.volume', access_limited=True)
   
    OP.set_inlets(pores=inlets)
    if not invasion:
        OP.set_outlets(pores=outlets)

    if not invasion:
        OP.run(points=n_pts)
    else:
        OP.run()

    return OP

def plot_cap_curve(perc_obj:op.algorithms.OrdinaryPercolation, ax=None, marker=None, show_break_line=False, label=None):

    p_break, sat_break = get_break_vals(perc_obj, return_sat=True)
    pc, sat = perc_obj.get_intrusion_data()

    if ax is None:
        fig, ax = plt.subplots()

    if marker is None:
        marker = '-'
    ax.plot(pc/1000, sat, marker, label=label)
    ax.grid()

    ax.set_xlabel('pressure (kPa)')
    ax.set_ylabel('invading phase saturation')
    ax.set_xlim(0, 25)
    if show_break_line:
        ax.axvline(x=p_break/1000, ymin=0, ymax=1.0, ls='--', c='r', label='break-pressure')

    ax.legend()

def get_diffusivity(phase:str, area, length, proj, conductance_model='throat.conduit_diffusive_conductance', boundaries=None):
    
    if boundaries is None:
        boundaries = ('left', 'right')
    
    in_ = boundaries[0]
    out_ = boundaries[1]
    phase = proj.phases()[phase]

    fd = op.algorithms.FickianDiffusion(project=proj)
    fd.setup(phase=phase)
    pn = proj.network
    inlets = pn.pores(in_)
    outlets = pn.pores(out_)
    fd.set_value_BC(pores=inlets, values=1)
    fd.set_value_BC(pores=outlets, values=0.5)

    # print('simulating fickian diffusion through the pore network...')
    fd.run()
    diff = fd.calc_effective_diffusivity(inlets=inlets, outlets=outlets, domain_area=area, domain_length=length)
    # print(diff)
    
    return diff[0] 

# def get_diffusivity(phase, area, length, proj):

#     phase = proj.phases()[phase]

#     fd = op.algorithms.FickianDiffusion(project=proj)
#     fd.setup(phase=phase)
#     pn = proj.network
#     inlets = pn.pores('left')
#     outlets = pn.pores('right')
#     fd.set_value_BC(pores=inlets, values=1)
#     fd.set_value_BC(pores=outlets, values=0.5)

#     print('simulating fickian diffusion through the pore network...')
#     fd.run()
    
#     return fd.calc_effective_diffusivity(inlets=inlets, outlets=outlets, domain_area=area, domain_length=length)[0]

def get_tomo_diffusivity(im_data, pore_phase_value, pixel_size):

    """Compute the effective diffusivity of a pore structure given 3D image dataset."""

    im = np.where(im_data==pore_phase_value, 1, 0)  # set pore phase value to 1
    im = np.array(im, dtype=bool) 
    
    # setup workspace        
    ws = op.Workspace()
    ws.settings["loglevel"] = 50
    ws.clear()
    proj = ws.new_project(name='proj')

    # extract network
    net = ps.networks.snow(im=im, voxel_size=pixel_size)
    op.io.PoreSpy.import_data(net, project=proj)

    # trim pores
    pore_health = proj.network.check_network_health()
    op.topotools.trim(network=proj.network, pores=pore_health['trim_pores'])

    # setup phase and physics
    air = op.phases.Air(project=proj, name='air')

    geo = proj.geometries()['geo_01']
    phy_air = op.physics.Standard(phase=air, geometry=geo, project=proj, name='phy_air')

    #use purcell model for pore entry pressure
    fiber_rad = 5e-6
    throat_diam = 'throat.diameter'
    pore_diam = 'pore.indiameter'
    pmod = op.models.physics.capillary_pressure.purcell
    phy_air.add_model(propname='throat.entry_pressure',
                        model=pmod,
                        r_toroid=fiber_rad,
                        diameter=throat_diam)

    # simulate Fickian diffusion
    shape = np.array(im.shape)
    area = shape[[1, 2]].prod() * (pixel_size**2)
    length = shape[0] * pixel_size

    diff = get_diffusivity(phase='air', area=area, length=length, proj=proj)

    return diff

def update_cap_sat_state(in_phase, def_phase, proj, sat_pt=None, pc_point=None, OP=None,  percolation_type=None, sim_sat=True):
    
    if sim_sat:
        if percolation_type == 'invasion':
            occupancy = OP.results(sat_pt)
        else:
            occupancy = OP.results(pc_point)
            
        in_phase['pore.occupancy'] = occupancy['pore.occupancy']
        in_phase['throat.occupancy'] = occupancy['throat.occupancy']

        def_phase['pore.occupancy'] = 1-occupancy['pore.occupancy']
        def_phase['throat.occupancy'] = 1-occupancy['throat.occupancy']
    else:
        p_occupancy = np.zeros(len(proj.network.pores()))
        t_occupancy = np.zeros(len(proj.network.throats()))

        in_phase['pore.occupancy'] = p_occupancy
        in_phase['throat.occupancy'] = t_occupancy

        def_phase['pore.occupancy'] = 1-p_occupancy
        def_phase['throat.occupancy'] = 1-t_occupancy

    mode = 'strict'
    
    phy_air = proj.physics()['phy_air']
    phy_water = proj.physics()['phy_water']
    
    phy_air.add_model(model=op.models.physics.multiphase.conduit_conductance,
                       propname='throat.conduit_diffusive_conductance',
                       throat_conductance='throat.diffusive_conductance',
                       mode=mode, factor=1e-3)
    
    phy_air.add_model(model=op.models.physics.multiphase.conduit_conductance,
                       propname='throat.conduit_hydraulic_conductance',
                       throat_conductance='throat.hydraulic_conductance',
                       mode=mode, factor=1e-3)
    
    phy_water.add_model(model=op.models.physics.multiphase.conduit_conductance,
                         propname='throat.conduit_diffusive_conductance',
                         throat_conductance='throat.diffusive_conductance',
                         mode=mode, factor=1e-3)
    
    phy_water.add_model(model=op.models.physics.multiphase.conduit_conductance,
                     propname='throat.conduit_hydraulic_conductance',
                     throat_conductance='throat.hydraulic_conductance',
                     mode=mode, factor=1e-3)
    
    props = ['throat.conduit_diffusive_conductance', 'throat.conduit_hydraulic_conductance']
    phy_air.regenerate_models(propnames=props)
    phy_water.regenerate_models(propnames=props)


def get_break_vals(alg_obj, return_sat=True):
    
    pcap, sat = alg_obj.get_intrusion_data()


    for i, p in enumerate(pcap):
        try:
            if alg_obj.is_percolating(p):
                p_break = p
                sat_break = sat[i]
                break
        except AttributeError:
            return None, None

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



def sample_cap(OP, sat_step, is_invasion=False ):
    
    # get saturation steps
    data = OP.get_intrusion_data()
    
    # check the type of percolation whether invasion or ordinary
    if is_invasion:
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

  
