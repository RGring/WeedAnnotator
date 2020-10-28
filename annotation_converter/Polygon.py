import numpy as np
class Polygon:
    def __init__(self, label):
        self.points = {"x": [], "y": []}
        self.label = label

    def get_polygon_points_as_array(self):
        return np.transpose(np.array([self.points["x"], self.points["y"]]))

    def get_polygon_points(self):
        return self.points

    def set_polygon_points_as_array(self, points):
        for point in points:
            self.points["x"].append(point[0][0])
            self.points["y"].append(point[0][1])

    def set_polygon_points(self, points):
        self.points = points

    def add_point(self, x, y):
        self.points["x"].append(x)
        self.points["y"].append(y)

    def get_label(self):
        return self.label