from fctest.__CVData__ import CVData

class GenericCV(CVData):
    def __init__(self, potential, current):
        super().__init__(potential, current)
    # def __init__(potential, current):
    #     super.__init__(potential)