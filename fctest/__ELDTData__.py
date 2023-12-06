
import matplotlib.pyplot as plt
import numpy as np

class ELDT:

    def __init__(self, elapsed_time, voltage, anode_pressure) -> None:
        self.elapsed_time = elapsed_time
        self.voltage = voltage
        self.anode_pressure=anode_pressure

    def plot_volt_over_time(self, label=None, color=None, return_axes=None, ax=None):

        if ax is None:
            ax = plt.gca()

        ax.plot(self.elapsed_time, self.voltage, '.-', c=color, label=label)

        ax.set_xlabel('elapsed time (s)')
        ax.set_ylabel('voltage (V)')
        ax.set_title('ELDT Test')
        ax.legend()
        ax.grid(True)

        if return_axes is not None and return_axes == True:
            plt.close()
            return ax
        
    def get_voltage_delta(self, n_pts:int=5, press_pt_buffer:int=10) -> float:

        highest_press_change_idx = np.argmax(np.diff(self.anode_pressure))

        voltage_pre_press = self.voltage[:highest_press_change_idx-press_pt_buffer]
        voltage_post_press = self.voltage[highest_press_change_idx+press_pt_buffer:]

        high_pts = voltage_pre_press[-n_pts:]
        low_pts = voltage_post_press[:n_pts]
        # top_n = voltage_pre_press[np.argpartition(-voltage_pre_press, n_pts)[:n_pts]]
        # bot_n = voltage_post_press[np.argpartition(voltage_post_press, n_pts)[:n_pts]]

        # delta_v = np.mean(top_n) - np.mean(bot_n)
        delta_v = np.mean(high_pts) -  np.mean(low_pts)

        return delta_v
    
    def get_max_voltage(self):

        return np.max(self.voltage)
    
    def get_min_voltage(self):

        max_idx = np.argmax(self.voltage)
        volt_region = self.voltage[max_idx:]

        return np.max(volt_region)