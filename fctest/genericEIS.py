import matplotlib.pyplot as plt
import numpy as np

class EISData:
    """
    A class representing the EIS Data type
    """

    def __init__(self, z_re, z_im, mea_area):
        self.z_re = z_re
        self.z_im = z_im

        self.mea_area = mea_area
        # todo: other parameters

    def plot_nyquist(self, y_error=None, x_error=None, fmt=None,  label=None, n_pts=None, unflip_im_axis=None, return_axis=None, ax=None, asr=None):
        
        z_im = self.z_im
        z_re = self.z_re
        
        if n_pts is not None:
            z_im = z_im[:n_pts]
            z_re = z_re[:n_pts]

        if y_error is not None:
            y_error = y_error[:n_pts]
        if x_error is not None:
            x_error = x_error[:n_pts]
        
        if ax is None:
            #fig, ax = plt.subplots()
            ax = plt.gca()
            
        if asr is None or asr == True:
            z_re = z_re * self.mea_area
            z_im = z_im * self.mea_area
            
            
            
            ax.set_xlabel('Re(Z) $\Omega cm^2$')            
            if unflip_im_axis == True:
                ax.set_ylabel('Im(Z) $\Omega cm^2$')
            else:
                z_im = -z_im
                ax.set_ylabel('$\minus$Im(Z) $\Omega cm^2$')
        else:
            
            ax.set_xlabel('Re(Z) $\Omega$')
            if unflip_im_axis == True:
                ax.set_ylabel('Im(Z) $\Omega$')
            else:
                z_im = -z_im
                ax.set_ylabel('$\minus$Im(Z) $\Omega$')
            
        if fmt is None:
            fmt = '.-'
        ax.errorbar(z_re, z_im, fmt=fmt, yerr=y_error, xerr=x_error, label=label)


        #ax.set_ylabel('$\minus$Im(Z) $\Omega$')
        ax.set_title('Nyquist plot')
        ax.legend()
        ax.grid()
        ax.axhline(y=0.0, alpha=0.08)


        if return_axis is not None and return_axis == True:
            plt.close()
            return ax

        


    def get_high_freq_intercept(self):
        idx = np.argmin(np.abs(self.z_im - 0))  # index closest to zero on im

        hfr = self.z_re_asr[idx]    

        return hfr

        # todo: I need to check if it is really and intercept 
        #       (crosses zero)