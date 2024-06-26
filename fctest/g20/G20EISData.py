from fctest.__EISData__ import EISData
import pandas as pd
import os
import warnings

class G20EISData(EISData):

    ENCODING = "ISO-8859-1"
    
    def __init__(self, data_path, mea_area=None, cell_name=None):

        path = os.path.normpath(data_path)
        raw_data = pd.read_csv(path, sep='\t', encoding=self.ENCODING)

        # split the relevant sections of the data
        data_section = raw_data.iloc[48:, 2:].reset_index(drop=True)
        col_names = raw_data.iloc[46, 2:].values
        units_section = raw_data.iloc[47, 2:].values
        data_section.columns = col_names

        test_date = pd.to_datetime(raw_data.iloc[2, 2] + ' ' + raw_data.iloc[3, 2])
        mea_area_from_file = float(raw_data.iloc[12, 2])
        if mea_area_from_file != mea_area:
            warnings.warn(f"file area is {mea_area_from_file}, while specified mea area is {mea_area}")
            
        initial_freq = float(raw_data.iloc[8, 2])
        final_freq = float(raw_data.iloc[9, 2])
        pts_per_decade = float(raw_data.iloc[10, 2])

        # relevant parameters
        freqs = pd.to_numeric(data_section.Freq).values
        z_mod = pd.to_numeric(data_section.Zmod).values
        z_real = pd.to_numeric(data_section.Zreal).values
        z_im = pd.to_numeric(data_section.Zimag).values
        z_ph = pd.to_numeric(data_section.Zphz).values


        super().__init__(z_re=z_real, z_im=z_im, mag=z_mod, freqs=freqs, mea_area=mea_area)

        # other properties specific to G20
        self.test_date = test_date
        self.initial_freq = initial_freq
        self.final_freq = final_freq
        self.points_per_decade = pts_per_decade
        self.file_name = os.path.basename(data_path)
        self.cell_name = cell_name



