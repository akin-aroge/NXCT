#from .__CVData__ import CVData
import pandas as pd
import os
from fctest.__CVData__ import CVData

class G20CVData(CVData):
    
    def __init__(self, data_path):

        raw_data = pd.read_csv(data_path, header=None)
        raw_data = pd.DataFrame(raw_data[0].str.split('\t').tolist())

        # parse parameters from raw_data
        test_date = pd.to_datetime(raw_data.iloc[3, 2] + \
            ' ' + raw_data.iloc[4, 2])
        mea_area = float(raw_data.iloc[17, 2])
        scan_rate = pd.to_numeric(raw_data.iloc[12, 2])
        n_cycles = pd.to_numeric(raw_data.iloc[14, 2])
        v_init = pd.to_numeric(raw_data.iloc[8, 2])
        v_limit_1 = pd.to_numeric(raw_data.iloc[9, 2])
        v_limit_2 = pd.to_numeric(raw_data.iloc[10, 2])
        v_final = pd.to_numeric(raw_data.iloc[11, 2])
        col_names = raw_data.iloc[63, :].values

        data_section = raw_data.iloc[65:, :]
        data_section.columns = col_names

        v_vs_ref = pd.to_numeric(data_section['Vf'].values)
        im = pd.to_numeric(data_section['Im'].values)

        super().__init__(potential=v_vs_ref, current=im)

        self.filename = os.path.basename(data_path)
        self.test_date = test_date

        self.mea_area = mea_area
        self.scan_rate = scan_rate
        self.n_cycles = n_cycles
        self.v_init = v_init
        self.v_limit_1 = v_limit_1
        self.v_limit_2 = v_limit_2
        self.v_final = v_final

        self.v_vs_ref = v_vs_ref
        self.im = im



