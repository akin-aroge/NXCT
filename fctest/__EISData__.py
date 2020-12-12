import matplotlib.pyplot as plt

class EISData:
    """
    A class representing the EIS Data type
    """

    def __init__(self, z_re, z_im, mag, freqs, mea_area):
        self.z_re = z_re
        self.z_im = z_im
        self.freqs = freqs
        self.mag = mag
        self.mea_area = mea_area
        # todo: other parameters

    def plot_nyquist(self, label=None, unflip_im_axis=None, return_axis=None, ax=None, asr=None):
        
        z_im = self.z_im
        z_re = self.z_re
        
        
        if ax is None:
            ax = plt.gca()
            
        if asr is None or asr == True:
            z_re = self.z_re * self.mea_area
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
            

 
        ax.plot(z_re, z_im, '-.', label=label)


        #ax.set_ylabel('$\minus$Im(Z) $\Omega$')
        ax.set_title('Nyquist plot')
        ax.legend()
        ax.grid()
        ax.axhline(y=0.0, alpha=0.08)


        if return_axis is not None and return_axis == True:
            plt.close()
            return ax

        

    def plot_bode(self, flip_im_axis=None, return_axis=None):
        
        fig, ax = plt.subplots()

        ax.semilogx(self.freqs, self.mag)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Magnitude')
        ax.set_title('Bode Plot')
        
        if return_axis is not None and return_axis == True:
            plt.close()
            return fig

    def get_high_freq_intercept(self):
        idx = self.freqs.argmax()

        return self.z_re[idx]

        # todo: I need to check if it is really and intercept 
        #       (crosses zero)