from fctest.__EISData__ import EISData
import pandas as pd
import os
import numpy as np

class SolatronEIS(EISData):

    
    def __init__(self, data_path, mea_area, cell_name=None):
        
        raw_data = pd.read_csv(data_path, sep='\t', skiprows=123, error_bad_lines=False)
        data_section = raw_data.drop(0, axis=0).reset_index(drop=True)

        # relevant parameters
        freqs = pd.to_numeric(data_section.iloc[:, 0].values)
        ac_amp = pd.to_numeric(data_section.iloc[0, 1])
        bias = pd.to_numeric(data_section.iloc[0, 2])
        z_re = pd.to_numeric(data_section.iloc[:, 4].values)
        z_im = pd.to_numeric(data_section.iloc[:, 5].values)

        z_ph = np.arctan2(z_re, z_im)
        z_mod = np.sqrt(z_re**2 + z_im**2)


        super().__init__(z_re=z_re, z_im=z_im, mag=z_mod, freqs=freqs, mea_area=mea_area)

        # other properties specific to Solatron
        self.ac_amp = ac_amp
        self.bias = bias

        self.file_name = os.path.basename(data_path)
        self.cell_name = cell_name
