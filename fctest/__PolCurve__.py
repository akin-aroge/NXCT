import matplotlib.pyplot as plt


class PolCurve:

    def __init__(self, current_density, voltage):

        self.current_density = current_density
        self.voltage = voltage

    def plot_pol_curve(self, label=None, return_axes=None, ax=None):

        if ax is None:
            ax = plt.gca()

        ax.plot(self.current_density, self.voltage, '-.', label=label)
        ax.set_xlabel('current density, j (A/$cm^2$)')
        ax.set_ylabel('voltage (V)')
        ax.set_title('polarisation curve')
        ax.legend()
        ax.grid(True)

        if return_axes is not None and return_axes == True:
            plt.close()
            return ax