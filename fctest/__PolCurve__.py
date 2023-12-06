from tkinter import Label
import matplotlib.pyplot as plt
import numpy as np


class PolCurve:

    def __init__(self, current_density, voltage):

        self.current_density = current_density
        self.voltage = voltage

    def plot_pol_curve(self, y_error=None, fmt=None, label=None, color=None, return_axes=None, ax=None):

        if ax is None:
            ax = plt.gca()

        if fmt is None:
            fmt = '.-'

        ax.errorbar(self.current_density, self.voltage, yerr=y_error, fmt=fmt, c=color, label=label)
        # ax.plot(self.current_density, self.voltage, marker='-', c=color, label=label)
        ax.set_xlabel('current density, j (A/$cm^2$)')
        ax.set_ylabel('voltage (V)')
        # ax.set_title('polarisation curve')
        ax.legend()
        ax.grid(True)

        if return_axes is not None and return_axes == True:
            plt.close()
            return ax

    def plot_pol_ir_adj(self, ohmic_asr, y_error=None, label=None, return_axes=None, ax=None):

        v_corr = self.voltage + self.current_density*ohmic_asr

        ax = self._plot_pol_curve(v_corr, self.current_density, \
        label=label, y_error=y_error, ax=ax)

        #ax.set_xlabel('current density, j (A/$cm^2$)')
        ax.set_ylabel('iR corrected voltage (V)')
        # ax.set_title('polarisation curve')
        # ax.legend()
        # ax.grid(True)

        if return_axes is not None and return_axes == True:
            plt.close()
            return ax

    def get_ocv(self):
        idx = 0
        ocv_0 = self.voltage[0]
        idx = np.argmin(np.abs(self.current_density - 0))  # index closest to zero
        if idx != 0:
            ocv_colsest_to_zero = self.voltage[idx]
            ocv = np.mean([ocv_0, ocv_colsest_to_zero])
        else:
            ocv = ocv_0
        

        return ocv

    def _plot_pol_curve(self, voltage, current_density, y_error=None,fmt=None, label=None, color=None,  ax=None):

        # if ax is None:
        #     ax = plt.gca()

        # ax.plot(current_density, voltage, '-.', label=label)
        # ax.set_xlabel('current density, j (A/$cm^2$)')
        # #ax.set_ylabel('voltage (V)')
        # # ax.set_title('polarisation curve')
        # ax.legend()
        # ax.grid(True)

        if ax is None:
            ax = plt.gca()

        if fmt is None:
            fmt = '.-'

        ax.errorbar(current_density, voltage, yerr=y_error, fmt=fmt, c=color, label=label)
        # ax.plot(self.current_density, self.voltage, marker='-', c=color, label=label)
        ax.set_xlabel('current density, j (A/$cm^2$)')
        ax.set_ylabel('voltage (V)')
        # ax.set_title('polarisation curve')
        ax.legend()
        ax.grid(True)

        # if return_axes is not None and return_axes == True:
        #     plt.close()
        #     return ax

        return ax
