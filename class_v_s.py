import numpy as np


def norm(a, b, c):
    n = np.sqrt(a * a + b * b + c * c)
    if n == 0:
        return np.array([0, 0, 0])
    else:
        return np.array([a / n, b / n, c / n])

class TPoint():
    def __init__(self, args, position, radius=0.5, end_point=None, num_end=0):
        self.position = position
        self.radius = radius
        self.parent = None
        self.child1 = None
        self.child2 = None
        self.rotate = None
        self.move = None
        self.deep = None
        self.num_end = num_end
        self.l = args.step


        if end_point == None:
            self.d = np.array([0, 0, 0])
        else:
            x_end = end_point.position[0]
            y_end = end_point.position[1]
            z_end = end_point.position[2]
            self.d = norm(x_end - self.position[0], y_end - self.position[1], z_end - self.position[2])

class Segment():
    def __init__(self, point_in, point_out):
        self.point_in = point_in
        self.point_out = point_out
        self.length = np.linalg.norm(point_in.position - point_out.position)
        self.radius = 0.5 + point_out.num_end * 0.03
        self.index = 0
        self.parent = None