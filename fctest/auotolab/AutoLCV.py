import pandas as pd
import os
from fctest.__CVData__ import CVData


class AutoLCV(CVData):

    def __init__(self, data_path):

        raw_data = pd.read_csv(data_path, sep='\t')
        time = raw_data.iloc[:, 2]
        potential = raw_data.iloc[:, 3]
        current = raw_data.iloc[:, 4]
        scan_num = raw_data.iloc[:, 0]

        super().__init__(potential, current)
        
        self.potential = potential
        self.current = current
        self.scan_num = scan_num
        self.file_name = os.path.basename(data_path)
        self.time = time
    