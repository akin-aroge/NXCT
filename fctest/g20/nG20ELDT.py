import pandas as pd
from fctest.__ELDTData__ import ELDT
import pathlib


class nG20ELDT(ELDT):

    def __init__(self, path) -> None:

        path = pathlib.Path(path)
        raw_data = pd.read_csv(path)
        data_part = raw_data.iloc[9:, :]
        data_part_df = pd.DataFrame(data_part[1:])
        data_part_df.columns = data_part.iloc[0, :].values

        log_rate = pd.to_numeric(raw_data.iloc[3, 1].split(" ")[0]) 
        # log_rate = raw_data.iloc[5, 1]
        test_date = pd.to_datetime(raw_data.iloc[5, 2])

        elapsed_time = pd.to_numeric(data_part_df['Elapsed time'].values)
        voltage = pd.to_numeric(data_part_df['voltage'].values)
        anode_pressure = pd.to_numeric(data_part_df['pressure_anode_inlet'].values)

        super().__init__(elapsed_time, voltage, anode_pressure=anode_pressure)
        self.test_date = test_date
        self.log_rate = log_rate
        self.filename = path.name