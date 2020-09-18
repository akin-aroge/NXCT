import matplotlib.pyplot as plt
class CVData:

    def __init__(self, potential, current):
        self.potential = potential
        self.current = current

    def plot_cv(self, ax=None, return_axes=None):
        
        if ax is None:
            ax = plt.gca()

        ax.plot(self.potential, self.current)
        ax.set_xlabel('voltage (V)')
        ax.set_ylabel('current (A)')
        ax.set_title('Cyclic Voltammetry')

        if return_axes is not None and return_axes == True:
            plt.close()
            return ax
