
import pandas as pd
import matplotlib.pyplot as pyplot
import os
from fctest.__PolCurve__ import PolCurve

class G20PolCurve(PolCurve):

    # mea_active_area = 0.21

    def __init__(self, path, mea_active_area):

        path = os.path.normpath(path)
        raw_data = pd.read_csv(path, sep='\t', header=None, engine='c')
        data_part = raw_data.iloc[5:,:][0].str.split(',', expand=True)
        test_date = pd.to_datetime(raw_data.iloc[1, 0])


        pol_data_part = pd.DataFrame(data_part[1:])
        pol_data_part.columns = data_part.iloc[0, :].values
        pol_data_part = pol_data_part.iloc[:, 4:]

        current = pol_data_part['current'].values
        voltage = pol_data_part['voltage'].values

        #data = pd.DataFrame(raw_data[0].str.split(',').tolist())
        # current = pd.to_numeric(data.iloc[6:, 4].values)
        # voltage = pd.to_numeric(data.iloc[6:, 7].values)
        #voltage_mean = pd.to_numeric(data.iloc[6:, 8])
        
        current_density = current / mea_active_area

        super().__init__(current_density, voltage)
        self.test_date = test_date
        self.mea_active_area = mea_active_area
        self.file_name = os.path.basename(path)





