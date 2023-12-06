import matplotlib.pyplot as plt
class CVData:

    def __init__(self, potential, current):
        self.potential = potential
        self.current = current

    def plot_cv(self, ax=None, label=None, return_axes=None):
        
        if ax is None:
            ax = plt.gca()

        ax.plot(self.potential, self.current, '.-', label=label)
        ax.set_xlabel('voltage (V)')
        ax.set_ylabel('current (A)')
        ax.set_title('Cyclic Voltammetry')
        ax.legend()

        if return_axes is not None and return_axes == True:
            plt.close()
            return ax
    
    def get_cdl(self, volt_range=(0.35, 0.65)):
        raise NotImplementedError
