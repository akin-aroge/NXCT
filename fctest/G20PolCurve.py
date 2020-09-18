import pandas as pd
import matplotlib.pyplot as pyplot
import os
from PolCurve import PolCurve

class G20PolCurve(PolCurve):

    # mea_active_area = 0.21

    def __init__(self, path, mea_active_area):

        path = os.path.normpath(path)

        raw_data = pd.read_csv(path, sep='\t', header=None)
        test_date = pd.to_datetime(raw_data.iloc[1, 0])
        data = pd.DataFrame(raw_data[0].str.split(',').tolist())
        current = pd.to_numeric(data.iloc[6:, 4].values)
        voltage = pd.to_numeric(data.iloc[6:, 7].values)
        #voltage_mean = pd.to_numeric(data.iloc[6:, 8])
        
        current_density = current / mea_active_area

        super().__init__(current_density, voltage)
        self.test_date = test_date
        self.mea_active_area = mea_active_area
        self.file_name = os.path.basename(path)


        