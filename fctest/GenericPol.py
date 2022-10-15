from fctest.__PolCurve__ import PolCurve


class GenericPol(PolCurve):
    def __init__(self, current_density, voltage):
        super().__init__(current_density=current_density, voltage=voltage)