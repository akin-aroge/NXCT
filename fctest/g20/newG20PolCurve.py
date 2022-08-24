import pandas as pd
import matplotlib.pyplot as pyplot
import os
from fctest.__PolCurve__ import PolCurve

class newG20PolCurve(PolCurve):

    # mea_active_area = 0.21

    def __init__(self, path, mea_active_area, cell_name=None, comment=None):

        path = os.path.normpath(path)

        raw_data = pd.read_csv(path, skiprows=9)
        test_date = pd.to_datetime(raw_data.iloc[1, 1])
        data = raw_data
        current = pd.to_numeric(data.iloc[:, 4].values)
        voltage = pd.to_numeric(data.iloc[:, 5].values)

        
        current_density = current / mea_active_area

        super().__init__(current_density, voltage)
        self.test_date = test_date
        self.mea_active_area = mea_active_area
        self.file_name = os.path.basename(path)
        self.cell_name = cell_name
        self.comment = comment