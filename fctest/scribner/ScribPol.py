from lib2to3.pgen2.parse import ParseError
import pandas as pd
import matplotlib.pyplot as pyplot
import os
from fctest.__PolCurve__ import PolCurve

class ScribPol(PolCurve):

    # mea_active_area = 0.21

    def __init__(self, path, mea_active_area, cell_name=None, comment=None, n_skip_rows=None):

        path = os.path.normpath(path)
        if n_skip_rows is None:
            #n_skip_rows = 41

            # TODO: refactor
            try:
                raw_data = pd.read_csv(path, sep='\t', skiprows=41)  # data hard to read without skiprows
            except pd.errors.ParserError:

                try:
                    raw_data = pd.read_csv(path, sep='\t', skiprows=57)  # data hard to read without skiprows
                except pd.errors.ParserError:
                    raw_data = pd.read_csv(path, sep='\t', skiprows=59)

        else:
            raw_data = pd.read_csv(path, sep='\t', skiprows=n_skip_rows) 




        data_part = raw_data.iloc[1:, [0, 1, 2, 5, 12, 13, 14, 17, 18]]
        data_part.columns = ['time', 'current', 'current_density', 'voltage', \
            'temp_cell', 'temp_anode', 'temp_cathode', 'rh_anode', 'rh_cathode']

        #current_density = pd.to_numeric(data_part.iloc[:, 2].values)
        current = pd.to_numeric(data_part.iloc[:, 1].values)
        current_density = current / mea_active_area
        voltage = pd.to_numeric(data_part.iloc[:, 3].values)    

        super().__init__(current_density, voltage)
        #self.test_date = test_date  
        self.mea_active_area = mea_active_area
        self.file_name = os.path.basename(path)
        self.cell_name = cell_name
        self.comment = comment
