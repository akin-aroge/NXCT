from fctest.__EISData__ import EISData
import pandas as pd
import os
import numpy as np

class AutoLEISData(EISData):

    ENCODING = "ISO-8859-1"
    
    def __init__(self, data_path, mea_area, cell_name=None):

        raw_data = pd.read_csv(data_path, sep='\t')
        raw_data = raw_data.iloc[:, 0].str.split(',', expand=True)

        data_section = raw_data.iloc[10:, :]
        data_section.columns = ['freq', 'ampl', 'bias', 'time', 'z_re', 'z_im', 'gd', 'err', 'range']

        # test_date = pd.to_datetime(raw_data.iloc[2, 2] + ' ' + raw_data.iloc[3, 2])
        # mea_area = float(raw_data.iloc[12, 2])
        # initial_freq = float(raw_data.iloc[8, 2])
        # final_freq = float(raw_data.iloc[9, 2])
        # pts_per_decade = float(raw_data.iloc[10, 2])

        # relevant parameters
        freqs = pd.to_numeric(data_section.freq).values
        z_real = pd.to_numeric(data_section.z_re).values
        z_im = pd.to_numeric(data_section.z_im).values
        z_mod = (z_real**2 + z_im**2)**(1/2)

        z_ph = np.arctan2(z_real, z_im)


        super().__init__(z_re=z_real, z_im=z_im, mag=z_mod, freqs=freqs, mea_area=mea_area)

        # other properties specific to G20
        #self.test_date = test_date
        #self.initial_freq = initial_freq
        #self.final_freq = final_freq
        #self.points_per_decade = pts_per_decade
        self.file_name = os.path.basename(data_path)
        self.cell_name = cell_name
