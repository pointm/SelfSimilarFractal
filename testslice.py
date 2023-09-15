from manim import *
from functools import reduce
import random
import itertools as it


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


class LindenmayerCurve(FractalCurve):
    """
        这段代码定义了一个类 LindenmayerCurve，它是 FractalCurve 的子类。
        LindenmayerCurve 类的作用是创建一个利用 L-系统（Lindenmayer system）生成的分形曲线，它有以下的属性和方法：

    - axiom：L-系统的公理，是一个字符串。公理是 L-系统的初始状态，它由一些字母组成，每个字母代表一个命令。
    - rule：L-系统的规则，是一个字典。规则是 L-系统的转换方式，它表示每个字母在下一步应该被替换成什么字符串。
    - scale_factor：分形曲线的缩放因子，是一个数值，默认为 2。这个数值表示每个子对象相对于原对象的缩放比例。
    - radius：分形曲线的半径，是一个数值，默认为 3。这个数值表示分形曲线的初始长度。
    - start_step：分形曲线的起始方向，是一个向量，默认为 RIGHT。这个向量表示分形曲线的第一段线段的方向。
    - angle：分形曲线的转角，是一个弧度值，默认为 pi/2。这个弧度值表示分形曲线在遇到 "+" 或 "-" 命令时应该旋转的角度。
    - expand_command_string：扩展命令字符串，是一个方法。这个方法接受一个命令字符串作为参数，返回一个扩展后的命令字符串。
    扩展的方式是对每个字母根据规则进行替换，如果没有对应的规则，就保留原字母。
    - get_command_string：获取命令字符串，是一个方法。
    这个方法根据 order 的值循环调用 expand_command_string 方法，得到最终的命令字符串。
    - get_anchor_points：获取分形曲线的角点，是一个方法。
    这个方法根据命令字符串和其他属性计算分形曲线的角点。计算的方式是从原点开始，根据起始方向和半径确定第一段线段的终点，并加入结果数组。
    然后对每个命令进行判断，如果是 "+" 或 "-"，就根据转角旋转当前方向；如果是其他字母，就沿着当前方向前进一段距离，并加入结果数组。
    最后返回结果数组减去它们的质心。:smile:
    """

    axiom = "A"
    rule = {}
    scale_factor = 2
    radius = 3
    start_step = RIGHT
    angle = np.pi / 2

    def expand_command_string(self, command):
        result = ""
        for letter in command:
            if letter in self.rule:
                result += self.rule[letter]
            else:
                result += letter
        return result

    def get_command_string(self):
        result = self.axiom
        for x in range(self.order):
            result = self.expand_command_string(result)
        return result

    def get_anchor_points(self):
        step = float(self.radius) * self.start_step
        step /= self.scale_factor**self.order
        curr = np.zeros(3)
        result = [curr]
        for letter in self.get_command_string():
            if letter == "+":
                step = rotate(step, self.angle)
            elif letter == "-":
                step = rotate(step, -self.angle)
            else:
                curr = curr + step
                result.append(curr)
        return np.array(result) - center_of_mass(result)


class FlowSnake(LindenmayerCurve):

    """
        这段代码定义了一个FlowSnake类，它是LindenmayerCurve类的子类。FlowSnake类用来生成一种叫做流蛇曲线的分形曲线，它有以下的属性和方法：

    - colors: 一个列表，表示曲线的颜色，可以是两种或者多种颜色。
    - axiom: 一个字符串，表示曲线的初始状态。
    - rule: 一个字典，表示曲线的生成规则，键是符号，值是替换后的字符串。
    - radius: 一个数字，表示曲线的半径。
    - scale_factor: 一个数字，表示曲线的缩放因子，这里是根号7。
    - start_step: 一个向量，表示曲线的初始方向，这里是右方。
    - angle: 一个数字，表示曲线的转角，这里是-1/3 * PI。

    - __init__(self, **kwargs): 一个方法，接受一些关键字参数作为输入，
    然后调用LindenmayerCurve类的构造函数来初始化对象。最后，根据曲线的阶数和旋转角度来旋转曲线，使其居中显示。
    """

    colors = [YELLOW, GREEN]
    axiom = "A"
    rule = {
        "A": "A-B--B+A++AA+B-",
        "B": "+A-BB--B-A++A+B",
    }
    radius = 6  # TODO, this is innaccurate
    scale_factor = np.sqrt(7)
    start_step = RIGHT
    angle = -np.pi / 3

    def __init__(self, **kwargs):
        LindenmayerCurve.__init__(self, **kwargs)
        self.rotate(-self.order * np.pi / 9, about_point=ORIGIN)


class TestFlowSnake(Scene):
    def construct(self):
        var = Line()
        curve = FlowSnake()
        self.play(Create(curve))
