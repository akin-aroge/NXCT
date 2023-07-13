import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.filters as flt
#%matplotlib inline
# since we can't use imports
import numpy as np
import scipy.ndimage.filters as flt
import warnings

from sklearn.mixture import GaussianMixture as GM

from scipy import interpolate, ndimage
from tqdm import tqdm

import skimage.measure as measure
from joblib import Parallel, delayed
from skimage.io import imread, imsave
import skimage as sk
import os


def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),sigma=0, option=1,ploton=False):
    """
    Anisotropic diffusion.

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                        2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration

    Returns:
            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference: 
    P. Perona and J. Malik. 
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi  
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.       
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """

    # ...you could always diffuse each color channel independently if you
    # really want

    print("performing 2D anisotropic filtering...")

    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep

        fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
        ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)

        ax1.imshow(img,interpolation='nearest')
        ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in tqdm(np.arange(1,niter)):

        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[:,:-1] = np.diff(imgout,axis=1)

        if 0<sigma:
            deltaSf=flt.gaussian_filter(deltaS,sigma)
            deltaEf=flt.gaussian_filter(deltaE,sigma)
        else: 
            deltaSf=deltaS
            deltaEf=deltaE

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaSf/kappa)**2.)/step[0]
            gE = np.exp(-(deltaEf/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaSf/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaEf/kappa)**2.)/step[1]

        # update matrices
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]

        # update the image
        imgout += gamma*(NS+EW)

        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)

    return imgout

def anisodiff3(stack,niter=1,kappa=50,gamma=0.1,step=(1.,1.,1.), sigma=0, option=1,ploton=False):
    """
    3D Anisotropic diffusion.

    Usage:
    stackout = anisodiff(stack, niter, kappa, gamma, option)

    Arguments:
            stack  - input stack
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (z,y,x)
            option - 1 Perona Malik diffusion equation No 1
                        2 Perona Malik diffusion equation No 2
            ploton - if True, the middle z-plane will be plotted on every 
                    iteration

    Returns:
            stackout   - diffused stack.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x,y and/or z axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference: 
    P. Perona and J. Malik. 
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi  
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.       
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """

    # ...you could always diffuse each color channel independently if you
    # really want

    print("performing 3D anisotropic filtering...")

    if stack.ndim == 4:
        warnings.warn("Only grayscale stacks allowed, converting to 3D matrix")
        stack = stack.mean(3)

    # initialize output array
    stack = stack.astype('float32')
    stackout = stack.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(stackout)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(stackout)
    gE = gS.copy()
    gD = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep

        showplane = stack.shape[0]//2

        fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
        ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)

        ax1.imshow(stack[showplane,...].squeeze(),interpolation='nearest')
        ih = ax2.imshow(stackout[showplane,...].squeeze(),interpolation='nearest',animated=True)
        ax1.set_title("Original stack (Z = %i)" %showplane)
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in tqdm(np.arange(1,niter)):

        # calculate the diffs
        deltaD[:-1,: ,:  ] = np.diff(stackout,axis=0)
        deltaS[:  ,:-1,: ] = np.diff(stackout,axis=1)
        deltaE[:  ,: ,:-1] = np.diff(stackout,axis=2)

        if 0<sigma:
            deltaDf=flt.gaussian_filter(deltaD,sigma)
            deltaSf=flt.gaussian_filter(deltaS,sigma)
            deltaEf=flt.gaussian_filter(deltaE,sigma)
        else:
            deltaDf=deltaD 
            deltaSf=deltaS
            deltaEf=deltaE
        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gD = np.exp(-(deltaD/kappa)**2.)/step[0]
            gS = np.exp(-(deltaS/kappa)**2.)/step[1]
            gE = np.exp(-(deltaE/kappa)**2.)/step[2]
        elif option == 2:
            gD = 1./(1.+(deltaD/kappa)**2.)/step[0]
            gS = 1./(1.+(deltaS/kappa)**2.)/step[1]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[2]

        # update matrices
        D = gD*deltaD
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'Up/North/West' by one
        # pixel. don't as questions. just do it. trust me.
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:,: ,: ] -= D[:-1,:  ,:  ]
        NS[: ,1:,: ] -= S[:  ,:-1,:  ]
        EW[: ,: ,1:] -= E[:  ,:  ,:-1]

        # update the image
        stackout += gamma*(UD+NS+EW)

        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(stackout[showplane,...].squeeze())
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)

    return stackout

    
def get_phase_frac(im, phase_val, ax):
    
    dim = im.shape
    
    if len(dim) == 2:
        im = im.reshape(-1, dim[0], dim[1])
        dim = im.shape
        
    im_ = np.where(im == phase_val, 1, 0)
    
    if ax == 'x':
        result = np.sum(im_, axis=0).sum(axis=0) / (dim[0] * dim[2])
        
    elif ax == 'y':
        result = np.sum(im_, axis=0).sum(axis=1) / (dim[0] * dim[1])
    
    elif ax == 'z':
        result = np.sum(im_, axis=1).sum(axis=1) / (dim[1] * dim[2])

    elif ax is None:
        result = np.mean(im_)
        
        
    return result

def get_saturation(im, air_phase_val, sat_phase_val, ax):

    sat_frac = get_phase_frac(im, phase_val=sat_phase_val, ax=ax)
    air_frac = get_phase_frac(im, phase_val=air_phase_val, ax=ax)
    saturation = sat_frac / (sat_frac + air_frac)

    return saturation

def get_z_aixs_profile(im_stack,  agg_func=None, phase_val=None):

    """returns an array where each item reprpesents a measure of the grayscale profile
    for the correspoiding slice
    """

    # if phase_val is None:
    #     phase_val = 255

    if phase_val is not None:
        im_stack = np.where(im_stack == phase_val, 1, 0)

    shape = im_stack.shape
    x = np.reshape(im_stack, (shape[0], shape[1]*shape[2]))

    if agg_func is None:
        agg_func = np.mean


    profile = agg_func(x, axis=1)

    return profile

def norm_stack(im_stack, normalizer=None, how=None):

    shape = im_stack.shape

    if normalizer is None:
        normalizer = im_stack[0]

    if how is None:
        how = "ratio"

    if how == "ratio":
        im_normed = im_stack / normalizer
    elif how == "diff":
        im_normed = im_stack - normalizer

    

    # im_normed_f = []
    # sigma=10
    # im_normed_f.extend(Parallel(n_jobs=5)(delayed(gauss_filter)(slc, sigma)  for slc in im_normed))
    im_normed_f = np.zeros_like(im_stack, dtype=np.float32)
    for slc_idx in np.arange(shape[0]):
        slc = im_normed[slc_idx]
        im_normed_f[slc_idx, :, :] = flt.gaussian_filter(slc, sigma=10)

    return im_normed_f
    


def get_porosity(im, phase_val):
    
    ps = np.mean(np.where(im==phase_val, 1, 0))

    return ps
    


def gaussian(x, mu, var):
    sig = var ** (1/2)
    return (1/(sig * (2*np.pi)**(1/2))) * np.exp(-(1/2) * ((x - mu)/sig)**2)
    #return np.exp(-(1/2) * ((x - mu)/sig)**2)

def gauss_fit1D(data, n_comp=1, comps=[''], bins=50, n_sample=100, title='', rand_state=42):
    """ gets intensity points and number of components and 
    plots an approximate gaussian fit of the different components"""



    gmm = GM(n_components=n_comp, covariance_type='full', tol=0.0001, random_state=rand_state).fit(data)

    vars_ = gmm.covariances_.flatten()
    stds = np.sqrt(vars_)
    mus = gmm.means_.flatten()
    ws = gmm.weights_.flatten()

    x_vals = np.linspace(np.min(data), np.max(data), num=n_sample)

    g_hats = np.zeros((len(x_vals), n_comp))
    for i in range(n_comp):
        g_hat = gaussian(x_vals, mus[i], vars_[i])
        g_hats[:, i] = g_hat

    # sum gaussians

    g_hats_total = np.dot(g_hats, ws)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.hist(data, bins=bins, density=True)

    for comp in range(n_comp):
        ax.plot(x_vals, ws[comp] * g_hats[:, comp], label="mu = %.2f,  std = %.2f, %s" % \
                (mus[comp], stds[comp], comps[comp]))


    ax.plot(x_vals, g_hats_total, label='fit')

    ax.set_xlabel('Grey scale value')
    ax.set_ylabel('density')
    ax.set_title(title, fontsize=19)
    ax.legend()

    return fig, ax, mus, stds, ws

def  get_cl_boundary(im, layer_to_keep='top', offset=0, connect=False):
    """ im: segmented catalyst outline image"""
    
    #cl_outline_im = imread(os.path.join(cl_outline_im_path))
    dim = im.shape
    
    # check what layer of the outline to keep, since cathode is at bottom, 
    # you want to keep the top layer
    if layer_to_keep == "bottom":
        
        #np.argmax returns the index of the first max which is the first 1
        bndry_slc = dim[0] - 1 -  np.argmax(im[::-1], axis=0)  #matrix
        bndry_slc = np.where(bndry_slc == dim[0] - 1, 0, bndry_slc)
        
    elif layer_to_keep == "top":
        bndry_slc = np.argmax(im, axis=0)
        
    # bndry_slc is a matrix with non zero items which represents the 
    # border point for the particular location e.g if (x, y) = 100 -> the 
    # border point for loc (x,y)  occurs on slice 100

    row, col = np.where(bndry_slc > 0)
    z = bndry_slc[row, col]
    
    
    # interp_smooth is 2D array where value of point (x, y) is the 
    # boundary slice number. This is also smoothened with filter        
    interp_im = np.zeros(dim, dtype=np.uint8)

    ny = np.arange(0, dim[1])
    nx = np.arange(0, dim[2])
    
    [xx,yy] = np.meshgrid(nx,ny) #grid to fit interpolation values
    


    if connect:
        interp = interpolate.griddata((col, row), z, (xx, yy), method='nearest')
        interp = ndimage.uniform_filter(interp, size=10) + offset

        interp_im[interp, yy, xx] = 255
    else:
        interp_im[z, row, col] = 255

    return interp_im


def fill_boundary(im, side_to_keep='top', offset=None):
    
    # filling up the relevat boundaries. For the setup, the cathode is up and the
    # anode is down. The filling strategy is to fill from slice 0 to top of cathode_cl
    # for cathode gdl and from last slice to bottom anode CL


    # create empty image, fill each slice with slice number and subtract interp-smooth.
    # Essentially: since interp_smooth holds the boundary slice number for each point x,y
    # we subtract it from each of the slices (which we fill with slice number) and if -ve
    # it implies it is below it, and if +ve it means the point is above it
    
    if offset is None:
        offset = 0

    #im = imread(os.path.join(cl_outline_im_path))
    dim = im.shape
    
    # check what layer of the outline to keep, since cathode is at bottom, 
    # you want to keep the top layer

    bndry_slc = np.argmax(im, axis=0) 

    if side_to_keep == 'top':
        bndry_slc = bndry_slc - offset
    if side_to_keep == 'bottom':
        bndry_slc = bndry_slc + offset        
    # bndry_slc is a matrix with non zero items which represents the 
    # border point for the particular location e.g if (x, y) = 100 -> the 
    # border point for loc (x,y)  occurs on slice 100

    del im

    gdl_mask = np.zeros(dim, dtype=np.uint8) 
    decision_im = gdl_mask \
                    + np.arange(dim[0]).reshape(-1, 1, 1) \
                    - bndry_slc

    del bndry_slc
    if side_to_keep == 'top':
        mz, mrow, mcol = np.where(decision_im < 0)  # select point before boundary
    elif side_to_keep == 'bottom':
        mz, mrow, mcol = np.where(decision_im > 0)  # select points below boundary
        
    gdl_mask[mz, mrow, mcol] = 255
    
    return gdl_mask   

def sep_cathode(ccseg_path, full_im_path, offset, trim=True):
    
    im = imread(full_im_path)
    cseg = imread(ccseg_path)

    im_dtype = str(im.dtype)

    # get boundary and mask
    gdl_bondry = get_cl_boundary(cseg, layer_to_keep='bottom', offset=offset, connect=True)  # bottom boundary of ccl
    c_gdl_mask = fill_boundary(gdl_bondry, side_to_keep='bottom')  # mask of gdl
    del gdl_bondry, cseg
    # select cathode gdl region
    im_cgdl = np.int32(np.int32(c_gdl_mask / 255) * np.int32(im)) 
    del c_gdl_mask, im 

    if trim:
    # trim original data to include just the cgdl region
        z, _, _ = np.nonzero(im_cgdl)  
        gdl_cut_point = np.min(z)  # slice of the topmost gdl point
        #im_cgdl = im_cgdl[gdl_cut_point:,:, :].copy()

        # the plus 2 is added so that the data siet matches the dry in dimension
        im_cgdl = im_cgdl[gdl_cut_point:,:, :].copy()
    
    return np.array(im_cgdl, dtype=im_dtype)



def gauss_filter(image, sigma):
    
    return sk.filters.gaussian(image, sigma, preserve_range=True)


def correct_illum(im, sigma=10, ref_region_spec=(15, 747, 100),  return_sc_field=None):
    """
    Corrects illumination across the stack of radiographs for quantitative information.
    The algorithm uses a reference patch which gives a sort of basis vector. 
    The relevant scaling  is obtained for the whole plane. 
    The basis vector is further decomposed into pattern, and trend component modelled by the change in GSV
    """
    shape = im.shape    
    im_G = []
    im_G.extend(Parallel(n_jobs=5)(delayed(gauss_filter)(slc, sigma)  for slc in im))
    im_G = np.array(im_G, dtype=np.float32)
    
    # pick the edge that would serve as reference where image does not change
    # in the through plane direction
    #l=15; r=150; t=747; b=1080
    l = ref_region_spec[0]; t = ref_region_spec[1]; w = ref_region_spec[2]
    r = l + w; b = t + w
    #ref_centre_idx = (l+(r-l)//2, t+(b-l)//2)
    im_G_ref_patch = im_G[:, t:b, l:r]    
    # compute the throughplane change
    ref_patch_mean = np.mean(im_G_ref_patch, axis=1).mean(axis=1)
    # subtract mean appears to do what I just want see calc in one note
    ref_patch_mean0 = ref_patch_mean - np.mean(ref_patch_mean)  
    im_G_mean0 = im_G  - np.mean(im_G, axis=0)
    del im_G

    # Derive trend to be removed from signal. This improves the computation of the scale field
    p = np.polyfit(np.arange(shape[0]), ref_patch_mean0, deg=1) # degree 1 works well for now
    trend = np.polyval(p, np.arange(shape[0]))
    trend_ = trend + np.abs(np.min(trend))
    ref_patch_detrend = ref_patch_mean0 - trend
    im_G_detrend = im_G_mean0 - trend.reshape(-1, 1, 1)
    del im_G_mean0
    ref_patch_norm = ref_patch_detrend / np.linalg.norm(ref_patch_detrend)
    
    im_scale_field = np.sum(ref_patch_norm.reshape(-1, 1, 1) * im_G_detrend, axis=0, dtype=np.float32)
    # recreate: we would use the pattern on the throu plane change at this reference and scale it accordingly
    # accross the plane
    im_correction = im_scale_field * np.float32(ref_patch_norm.reshape(-1, 1, 1)) + np.float32(trend.reshape(-1, 1, 1))
    #del im_scale_field
    im_corrected = im - im_correction
    del im_correction

    if return_sc_field:
        return im_corrected, im_scale_field
    else:

        return im_corrected#, im_scale_field


def correct_illum_m2(im, sigma=5, ref_region_spec=(15, 747, 100), return_sc_field=None):
    #im_G = np.zeros_like(im)
#     for i, slc in enumerate(im):

#         im_G[i] = sk.filters.gaussian(slc, sigma=sigma, preserve_range=True)
    shape = im.shape    
    im_G = []
    im_G.extend(Parallel(n_jobs=5)(delayed(gauss_filter)(slc, sigma)  for slc in im))
    im_G = np.float32(np.array(im_G))
    
    # pick the edge that would serve as reference where image does not change
    # in the through plane direction
    #l=15; r=150; t=747; b=1080
    l = ref_region_spec[0]; t = ref_region_spec[1]; w = ref_region_spec[2]
    r = l + w; b = t + w
    ref_centre_idx = (l+(r-l)//2, t+(b-l)//2)
    im_G_ref_patch = im_G[:, t:b, l:r].copy()   

    # compute the throughplane change
    ref_patch_mean = np.mean(im_G_ref_patch, axis=1).mean(axis=1)
    # subtract mean appears to do what I just want see calc in one note
    ref_patch_mean = ref_patch_mean - np.mean(ref_patch_mean)   

    # we would use the pattern on the throu plane change at this reference and scale it accordingly
    # accross the plane

    ref_patch_norm = ref_patch_mean / np.linalg.norm(ref_patch_mean)
    im_scale_field = np.sum(ref_patch_norm.reshape(-1, 1, 1) * im_G, axis=0)

    im_correction = im_scale_field * ref_patch_norm.reshape(-1, 1, 1)

    #im_corrected = im - im_correction + im[0]  # this one takes the image to typical values
    im_corrected = im - im_correction
    
    if return_sc_field:
        return im_corrected, im_scale_field
    else:

        return im_corrected#, im_scale_field


def correct_illum_m3(im, sigma=250, ref_region_spec=(15, 747, 100), return_sc_field=None):
    #im_G = np.zeros_like(im)
#     for i, slc in enumerate(im):

#         im_G[i] = sk.filters.gaussian(slc, sigma=sigma, preserve_range=True)
    shape = im.shape    
    im_G = []
    im_G.extend(Parallel(n_jobs=5)(delayed(gauss_filter)(slc, sigma)  for slc in im))
    im_G = np.float32(np.array(im_G))
    
    # pick the edge that would serve as reference where image does not change
    # in the through plane direction
    l = ref_region_spec[0]; t = ref_region_spec[1]; w = ref_region_spec[2]
    r = l + w; b = t + w
    #l=15; r=150; t=747; b=1080
    crop_centre_idx = (l+(r-l)//2, t+(b-l)//2)
    im_G_edge_crop = im_G[:, t:b, l:r]    

    # compute the throughplane change
    z_sum = np.mean(im_G_edge_crop, axis=1).mean(axis=1)
    z_sum_diff = np.diff(z_sum)

    # we would use the patter on the throu plane change at this reference and scale it accordingly
    # accross the plane

    # first pick a notable point in the through-plane direction: that max
    max_delta_idx = np.argmax(z_sum_diff)  # index of max change

    # find the corresponding changes aacross the plane
    im_max_diff  = im_G[max_delta_idx+1] - im_G[max_delta_idx]

    # find the scaling factor: how that point scales across the field
    # this scaling will be used to generate other points from the reference through-plane change 
    im_scale_field = im_max_diff / z_sum_diff[max_delta_idx]
    #im_scale_field = im_max_diff / im_max_diff[crop_centre_idx[1], crop_centre_idx[0]]

    # generate full stack changes and include initial zero
    im_correction_delta = im_scale_field * z_sum_diff.reshape(-1, 1, 1)
    im_correction_delta = np.insert(im_correction_delta, 0, 
                                    np.zeros((shape[1],shape[2])), axis=0)

    # the corrected image will be the raw image with the cumulative sum removed
    im_corrected = im - np.cumsum(im_correction_delta, axis=0)
    
    if return_sc_field:
        return im_corrected, im_scale_field
    else:

        return im_corrected#, im_scale_field



def filter_particle_area(label, thresh):
    """ 
    return: f_label - particle_size filtered image according to thresh and each pixel now has the component size
    """
    f_label = label.copy()
    
    ms = measure.regionprops(label)

    for region in ms:
        if region.area < thresh:
            f_label[f_label == region.label ] = 0
        else:
            f_label[f_label == region.label] = region.area
    
        
    return f_label

def crop_from_centre(x, width):
    
    mid_x = x.shape[1] // 2
    mid_y = x.shape[2] // 2 
    
    crop = x[:, mid_x-(width//2):mid_x+(width//2), mid_y-(width//2):mid_y+(width//2)]
    
    return crop

def save_figure(fname, obj, figdir=None):
    """wrapper to save figure in desied directory"""
    if figdir is None:
        figdir = r"..\\..\\..\\..\\..\\..\sfu\phd\research_in_motion\development\write-up\misc"
    path = os.path.join(os.path.normpath(figdir), fname)
    obj.savefig(path, bbox_inches='tight')



def save_image(fname, obj, save_dir=None):
    """wrapper to save figure in desied directory"""
    if save_dir is None:
        figdir = r""
    path = os.path.join(os.path.normpath(save_dir), fname)
    imsave(path, obj)