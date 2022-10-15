from tkinter import Label
import matplotlib.pyplot as plt


class PolCurve:

    def __init__(self, current_density, voltage):

        self.current_density = current_density
        self.voltage = voltage

    def plot_pol_curve(self, label=None, return_axes=None, ax=None):

        if ax is None:
            ax = plt.gca()

        ax.plot(self.current_density, self.voltage, '.-', label=label)
        ax.set_xlabel('current density, j (A/$cm^2$)')
        ax.set_ylabel('voltage (V)')
        ax.set_title('polarisation curve')
        ax.legend()
        ax.grid(True)

        if return_axes is not None and return_axes == True:
            plt.close()
            return ax

    def plot_pol_ir_adj(self, ohmic_asr, label=None, return_axes=None, ax=None):

        v_corr = self.voltage + self.current_density*ohmic_asr

        ax = self._plot_pol_curve(v_corr, self.current_density, \
        label=label, ax=ax)

        #ax.set_xlabel('current density, j (A/$cm^2$)')
        ax.set_ylabel('iR corrected voltage (V)')
        # ax.set_title('polarisation curve')
        # ax.legend()
        # ax.grid(True)

        if return_axes is not None and return_axes == True:
            plt.close()
            return ax


    def _plot_pol_curve(self, voltage, current_density, label=None,  ax=None):

        if ax is None:
            ax = plt.gca()

        ax.plot(current_density, voltage, '-.', label=label)
        ax.set_xlabel('current density, j (A/$cm^2$)')
        #ax.set_ylabel('voltage (V)')
        ax.set_title('polarisation curve')
        ax.legend()
        ax.grid(True)

        return ax
