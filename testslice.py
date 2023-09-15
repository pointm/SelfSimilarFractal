from manim import *
from functools import reduce
import random
import itertools as it


from manim import *

class SquareExample(Scene):
    def construct(self):
        # 创建一个 VMobject
        vmob = VMobject()
        # 将四个点设置为角点，形成一个正方形
        vmob.set_points_as_corners([UP, RIGHT, DOWN, LEFT, UP])
        # 将 VMobject 添加到场景中
        self.play(Create(vmob))



def rotate(points, angle=np.pi, axis=OUT):
    """
    旋转函数
    输入:
    --points 传入的被旋转的点的坐标（三维）
    --angle 旋转的角度,默认为PI弧度/180°
    --axis 被指定的旋转的轴的方向(默认为z轴,也就是指向屏幕穿出的轴)
    输出:
    --points 被旋转之后的点的坐标
    传入一个三维的列表points,代表坐标。之后针对这个坐标绕着指定axis方向
    的轴进行角度为angle的旋转
    这种计算旋转的方式被称为罗德里格斯方式
    """
    if axis is None:
        return points  # 如果被指定的方向无效, 那么就不对点进行旋转
    matrix = rotation_matrix(angle, axis)
    points = np.dot(points, np.transpose(matrix))
    return points


class JaggedCurvePiece(VMobject):
    """
    类 JaggedCurvePiece是 VMobject 的子类。

    这其中有一个方法insert_n_curves
    - 如果曲线片段没有任何曲线，就设置它的点为一个零向量。
    - 获取曲线片段的锚点，即每个曲线的起点和终点。
    - 在锚点数组中，根据 n 的值，均匀地选取 n + len(anchors) 个索引。
    - 用选取的锚点作为新的角点，重新设置曲线片段的点。

    这样，就可以在原来的曲线片段中插入 n 个新的曲线，使得曲线片段看起来更加锯齿化。
    """

    def insert_n_curves(self, n):
        if self.get_num_curves() == 0:
            self.set_points(np.zeros((1, 3)))  # 没有点的话就返回一个0向量
        anchors = self.get_anchors()
        indices = np.linspace(0, len(anchors) - 1, n + len(anchors)).astype("int")
        self.set_points_as_corners(anchors[indices])
        """set_points_as_corners的意思是：
            - 给定一个点的数组，把它们设置为VMobject的角点。
            - 为了实现这个目的，这个算法会调整每个曲线的控制点，使得它们与锚点对齐，从而使得结果的贝塞尔曲线就是两个锚点之间的线段。
            - 参数是一个点的数组，它们会被设置为角点。
            - 返回值是自身，也就是VMobject对象。

                简单地说，类里面的insert_n_curves这个方法可以把一个曲线变得更加锯齿化
            """


class FractalCurve(VMobject):
    """
    这段代码定义了一个类 FractalCurve，它是 VMobject 的子类。FractalCurve 类的作用是创建一个分形曲线，它有以下的属性和方法：

    - radius：分形曲线的半径，是一个数值，默认为 3。
    - order：分形曲线的阶数，是一个整数，默认为 5。阶数越高，分形曲线越复杂。
    - colors：分形曲线的颜色，是一个颜色列表，默认为 [RED, GREEN]。分形曲线会根据这个列表生成一个渐变色。
    - num_submobjects：分形曲线的子对象数量，是一个整数，默认为 20。
    子对象是指分形曲线中的每一段锯齿形的曲线片段，它们是 JaggedCurvePiece 类的实例。
    - monochromatic：分形曲线是否为单色，是一个布尔值，默认为 False。
    如果为 True，分形曲线不会使用渐变色，而是使用 VMobject 的默认颜色。
    - order_to_stroke_width_map：分形曲线的阶数和描边宽度的映射，是一个字典，默认为 {3: 3, 4: 2, 5: 1}。
    这个字典表示不同阶数的分形曲线应该使用的描边宽度，如果阶数超过字典中的最大键值，就使用最大键值对应的描边宽度。
    - init_points：初始化分形曲线的点，是一个方法。
    这个方法会调用 get_anchor_points 方法获取分形曲线的角点，然后调用 set_points_as_corners
    方法把角点设置为 VMobject 的角点。

    - init_colors：初始化分形曲线的颜色，是一个方法。这个方法会调用 VMobject 的 init_colors 方法，
    并根据 colors 的值设置分形曲线的渐变色。然后根据 order 和 order_to_stroke_width_map 的值设置分形曲线的描边宽度。
    - get_anchor_points：获取分形曲线的角点，是一个方法。这个方法没有实现，需要在子类中重写。
    """

    radius = 3
    order = 5
    colors = [RED, GREEN]
    num_submobjects = 20
    monochromatic = False
    order_to_stroke_width_map = {
        3: 3,
        4: 2,
        5: 1,
    }

    def init_points(self):
        points = self.get_anchor_points()
        self.set_points_as_corners(points)
        if not self.monochromatic:
            alphas = np.linspace(0, 1, self.num_submobjects)
            for alpha_pair in zip(alphas, alphas[1:]):
                submobject = JaggedCurvePiece()
                submobject.pointwise_become_partial(self, *alpha_pair)
                self.add(submobject)
            self.set_points(np.zeros((0, 3)))

    def init_colors(self):
        VMobject.init_colors(self)
        self.set_color_by_gradient(*self.colors)
        for order in sorted(self.order_to_stroke_width_map.keys()):
            if self.order >= order:
                self.set_stroke(width=self.order_to_stroke_width_map[order])

    def get_anchor_points(self):
        raise Exception("Not implemented")


class SelfSimilarSpaceFillingCurve(FractalCurve):
    offsets = []
    # keys must awkwardly be in string form...
    offset_to_rotation_axis = {}
    scale_factor = 2
    radius_scale_factor = 0.5

    def transform(self, points, offset):
        """
        How to transform the copy of points shifted by
        offset.  Generally meant to be extended in subclasses
        """
        copy = np.array(points)
        if str(offset) in self.offset_to_rotation_axis:
            copy = rotate(copy, axis=self.offset_to_rotation_axis[str(offset)])
        copy /= (self.scale_factor,)
        copy += offset * self.radius * self.radius_scale_factor
        return copy

    def refine_into_subparts(self, points):
        transformed_copies = [self.transform(points, offset) for offset in self.offsets]
        return reduce(lambda a, b: np.append(a, b, axis=0), transformed_copies)

    def get_anchor_points(self):
        points = np.zeros((1, 3))
        for count in range(self.order):
            points = self.refine_into_subparts(points)
        return points

    def generate_grid(self):
        raise Exception("Not implemented")


class HilbertCurve(SelfSimilarSpaceFillingCurve):
    offsets = [
        LEFT + DOWN,
        LEFT + UP,
        RIGHT + UP,
        RIGHT + DOWN,
    ]
    offset_to_rotation_axis = {
        str(LEFT + DOWN): RIGHT + UP,
        str(RIGHT + DOWN): RIGHT + DOWN,
    }


class TestSlice(Scene):
    def construct(self):
        # 创建一个圆形的曲线片段
        curve = JaggedCurvePiece()
        curve.set_points_as_corners([UP*2, LEFT, DOWN, RIGHT, ORIGIN])
        # 在曲线片段中插入 4 个新的曲线
        c1 = curve.insert_n_curves(6)
        c2 = curve.insert_n_curves(3)
        self.add(curve)
        # for i in range(4):
        self.play(Transform(c2, c1))
        