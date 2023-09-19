from manim import *

class EllipseMobject(VMobject):
    def __init__(self, center, width, height):
        super().__init__()
        self.center = center
        self.width = width
        self.height = height

    def get_bounding_box(self):
        return Rectangle(
            width=self.width, height=self.height,
            center=self.center
        ).get_bounding_box()

    def get_points(self):
        xs = [self.center[0] - self.width / 2, self.center[0] + self.width / 2]
        ys = [self.center[1] - self.height / 2, self.center[1] + self.height / 2]



class MyScene(Scene):
    def construct(self):
        shape = EllipseMobject((0, 0), 100, 50)
        self.add(shape)
        self.wait()
