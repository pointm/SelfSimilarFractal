from time import sleep
from manim import *
from functools import reduce
import random
import itertools as it


from manim import *

# class SquareExample(Scene):
#     def construct(self):
#         # 创建一个 VMobject
#         vmob = VMobject()
#         # 将四个点设置为角点，形成一个正方形
#         vmob.set_points_as_corners([UP, RIGHT, DOWN, LEFT, UP])
#         # 将 VMobject 添加到场景中
#         self.play(Create(vmob))


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

    这段代码定义了一个名为JaggedCurvePiece的类，它是VMobject的子类。这个类有一个方法insert_n_curves，
    其功能是在曲线片段中插入新的曲线，使其看起来更加锯齿化。
    insert_n_curves方法的功能如下：

    如果曲线片段没有任何曲线（即get_num_curves()返回0），它将会设置曲线片段的点为一个零向量（即set_points(np.zeros((1, 3)))）。
    它获取曲线片段的锚点（即每个曲线的起点和终点），这些锚点存储在anchors数组中。
    然后，根据给定的n值和anchors数组的长度，它均匀地选取n + len(anchors)个索引（使用np.linspace函数实现）。
    最后，它用选取的锚点作为新的角点，重新设置曲线片段的点（即set_points_as_corners(anchors[indices])）。

    总的来说，这个方法通过在原始曲线片段中插入新的曲线，使得曲线片段看起来更加锯齿化。
    """

    def insert_n_curves(self, n):
        # 如果当前对象的曲线数量为0，则执行以下操作
        if self.get_num_curves() == 0:
            # 调用set_points方法将当前对象的点集设置为一个只包含原点的数组
            self.set_points(np.zeros((1, 3)))  # 没有点的话就返回一个0向量
        # 调用get_anchors方法获取当前对象的锚点，也就是曲线上的顶点
        anchors = self.get_anchors()
        # 使用np.linspace函数生成一个等差数列，长度为n加上锚点的数量，然后转换为整数类型这一行是生成一个等差数列，数列的长度为n加上锚点的数量，
        # 数列的元素为从0到锚点数量减一的整数，这些整数表示锚点的索引。
        # 例如，如果锚点的数量为4，n为2，那么数列就是[0, 1, 2, 3, 4, 5]，其中0, 2, 4分别对应原有的第一个，第二个，第三个锚点，
        # 而1, 3, 5则对应新添加的锚点。
        indices = np.linspace(0, len(anchors) - 1, n + len(anchors)).astype("int")
        # 调用set_points_as_corners方法将点集设置为角点，也就是将锚点按照数列中的索引连接成一条折线
        # 这一行是将锚点按照数列中的索引连接成一条折线，也就是将原有的锚点和新添加的锚点按照顺序连接起来。
        # 例如，如果原有的锚点是A, B, C，新添加的锚点是D, E, F，那么折线就是A-D-B-E-C-F。这样就实现了在当前对象中插入n条曲线的效果。
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

    # 定义一个实例方法，用于初始化点集，这是VMobject类的一个重写方法
    def init_points(self):
        # 调用get_anchor_points方法获取锚点，也就是曲线上的顶点
        points = self.get_anchor_points()
        # 调用set_points_as_corners方法将点集设置为角点，也就是将点集连接成一条折线
        self.set_points_as_corners(points)
        # 如果曲线不是单色的，则执行以下操作
        if not self.monochromatic:
            # 使用np.linspace函数生成一个等差数列，长度为子对象的数量
            alphas = np.linspace(0, 1, self.num_submobjects)
            # 对于数列中相邻的两个元素，执行以下操作
            for alpha_pair in zip(alphas, alphas[1:]):
                # 创建一个JaggedCurvePiece对象，这是一个表示锯齿形曲线片段的类
                submobject = JaggedCurvePiece()  # 实例化出现
                # 调用pointwise_become_partial方法将子对象设置为曲线的一部分，参数为数列中的两个元素，表示起始和结束位置
                submobject.pointwise_become_partial(self, *alpha_pair)
                # 调用add方法将子对象添加到当前对象中
                self.add(submobject)
            # 调用set_points方法将当前对象的点集设置为空数组，也就是清空点集
            self.set_points(np.zeros((0, 3)))

    # 定义一个实例方法，用于初始化颜色，这是VMobject类的一个重写方法
    def init_colors(self):
        # 调用VMobject类的init_colors方法进行基本的颜色初始化
        VMobject.init_colors(self)
        # 调用set_color_by_gradient方法将当前对象按照颜色渐变进行着色
        self.set_color_by_gradient(*self.colors)
        # 对于描边宽度字典中已排序的键（也就是阶数），执行以下操作
        for order in sorted(self.order_to_stroke_width_map.keys()):
            # 如果当前对象的阶数大于等于键值，则执行以下操作
            if self.order >= order:
                # 调用set_stroke方法将当前对象的描边宽度设置为字典中对应的值
                self.set_stroke(width=self.order_to_stroke_width_map[order])

    # 定义一个实例方法，用于获取锚点，这是一个抽象方法，需要在子类中实现
    def get_anchor_points(self):
        raise Exception("Not implemented")


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


class SelfSimilarSpaceFillingCurve(FractalCurve):
    offsets = []  # 定义一个类属性，存储每个偏移量对应的旋转轴，键必须是字符串形式
    # keys must awkwardly be in string form...
    offset_to_rotation_axis = {}  # 定义一个类属性，存储缩放因子，用于控制子部分的大小
    scale_factor = 2  # 定义一个类属性，存储半径缩放因子，用于控制子部分的位置
    radius_scale_factor = 0.5

    # 定义一个实例方法，用于对点集进行变换，接受两个参数：points和offset
    def transform(self, points, offset):
        """
        How to transform the copy of points shifted by
        offset.  Generally meant to be extended in subclasses
        """
        # 创建一个点集的副本
        copy = np.array(points)
        # 如果偏移量在旋转轴字典中有对应的值，则对副本进行旋转变换，旋转轴为对应的值
        if str(offset) in self.offset_to_rotation_axis:
            copy = rotate(copy, axis=self.offset_to_rotation_axis[str(offset)])
        # 对副本进行缩放变换，缩放因子为类属性scale_factor的倒数
        copy /= (self.scale_factor,)
        # 对副本进行平移变换，平移向量为偏移量乘以半径乘以半径缩放因子
        copy += offset * self.radius * self.radius_scale_factor
        # 返回变换后的副本
        return copy

    # 定义一个实例方法，用于将点集细化为子部分，接受一个参数：points
    def refine_into_subparts(self, points):
        # 对每个偏移量，调用transform方法对点集进行变换，并将结果存入一个列表
        transformed_copies = [self.transform(points, offset) for offset in self.offsets]
        # 使用reduce函数将列表中的所有点集合并为一个数组，并返回
        return reduce(lambda a, b: np.append(a, b, axis=0), transformed_copies)

    # 定义一个实例方法，用于获取锚点，也就是曲线上的顶点
    def get_anchor_points(self):
        # 创建一个只包含原点的数组
        points = np.zeros((1, 3))
        # 对于每个阶数，调用refine_into_subparts方法将点集细化为子部分，并更新points
        for count in range(self.order):
            points = self.refine_into_subparts(points)
        # 返回最终的点集
        return points

    # 定义一个实例方法，用于生成网格，这是一个抽象方法，需要在子类中实现
    def generate_grid(self):
        raise Exception("Not implemented")


"""
这段代码是用来定义一个自相似空间填充曲线的类的，它继承了FractalCurve类。这个类有以下几个属性和方法：

- offsets: 一个列表，存储了每个子曲线相对于原始曲线的偏移量。
- offset_to_rotation_axis: 一个字典，存储了每个偏移量对应的旋转轴，用来旋转子曲线。键必须是字符串形式的偏移量。
- scale_factor: 一个数字，表示子曲线相对于原始曲线的缩放比例。
- radius_scale_factor: 一个数字，表示子曲线相对于原始曲线的半径缩放比例。
- transform(self, points, offset): 一个方法，用来对原始曲线的点进行变换，生成子曲线的点。
它接受两个参数：points是一个numpy数组，表示原始曲线的点；offset是一个三维向量，表示偏移量。
它首先检查偏移量是否在offset_to_rotation_axis中，如果是，则对points进行旋转；然后对points进行缩放；最后加上偏移量乘以半径和半径缩放比例。
返回变换后的点。
- refine_into_subparts(self, points): 一个方法，用来把原始曲线的点细化成多个子曲线的点。
它接受一个参数：points是一个numpy数组，表示原始曲线的点。它遍历offsets中的每个偏移量，调用transform方法生成子曲线的点，并把它们拼接起来。
返回拼接后的点。
- get_anchor_points(self): 一个方法，用来获取曲线的锚点。
它不接受任何参数。它首先创建一个零向量作为初始点，然后根据order属性（从FractalCurve类继承）重复调用refine_into_subparts方法，每次细化一次。
返回最终细化后的点。
- generate_grid(self): 一个方法，用来生成网格。它不接受任何参数。它抛出一个异常，表示没有实现。这个方法应该在子类中重写。

: [FractalCurve](https://github.com/3b1b/manim/blob/master/manimlib/mobject/fractals.py)
"""


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

class PeanoCurve(SelfSimilarSpaceFillingCurve):
    colors = [PURPLE, TEAL]
    offsets = [
        LEFT + DOWN,
        LEFT,
        LEFT + UP,
        UP,
        ORIGIN,
        DOWN,
        RIGHT + DOWN,
        RIGHT,
        RIGHT + UP,
    ]
    offset_to_rotation_axis = {
        str(LEFT): UP,
        str(UP): RIGHT,
        str(ORIGIN): LEFT + UP,
        str(DOWN): RIGHT,
        str(RIGHT): UP,
    }
    scale_factor = 3
    radius_scale_factor = 2.0 / 3


class TestHilbertCurve(Scene):
    def construct(self):
        hilbert_curve = HilbertCurve()

        hilbert_group = []
        num = 4

        for i in range(num):
            hilbert_curve.order = i + 1
            points = hilbert_curve.get_anchor_points()
            hilbert_curve.set_points_as_corners(points)
            hilbert_curve.colors = [RED, GREEN]
            self.play(Create(hilbert_curve), run_time=i+1)
            self.wait()
            hilbert_group.append(hilbert_curve)

class TestPeanoCurve(Scene):
    def construct(self):
        hilbert_curve = PeanoCurve()

        num = 4

        for i in range(num):
            hilbert_curve.order = i + 1
            points = hilbert_curve.get_anchor_points()
            hilbert_curve.set_points_as_corners(points)
            hilbert_curve.set_color([RED, GREEN])
            self.play(Create(hilbert_curve), run_time=i+1)
            self.wait()
          
