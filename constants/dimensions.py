class BadmintonCourtDimensions:
    def __init__(self):
        self.HEIGHT = 13.40
        self.WIDTH = 6.1

    def get_dimension_coordinates(self):
        return [0, 0, self.WIDTH, 0, 0, self.HEIGHT, self.WIDTH, self.HEIGHT]