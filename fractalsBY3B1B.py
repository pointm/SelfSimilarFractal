from functools import reduce
import random
from manim import *
import itertools as it


# from manimlib.constants import *
# # from manimlib.for_3b1b_videos.pi_creature import PiCreature
# # from manimlib.for_3b1b_videos.pi_creature import Randolph
# # from manimlib.for_3b1b_videos.pi_creature import get_all_pi_creature_modes
# from manimlib.mobject.geometry import Circle
# from manimlib.mobject.geometry import Polygon
# from manimlib.mobject.geometry import RegularPolygon
# from manimlib.mobject.types.vectorized_mobject import VGroup
# from manimlib.mobject.types.vectorized_mobject import VMobject
# from manimlib.utils.bezier import interpolate
# from manimlib.utils.color import color_gradient
# # from manimlib.utils.dict_ops import digest_config
# from manimlib.utils.space_ops import center_of_mass
# from manimlib.utils.space_ops import compass_directions
# from manimlib.utils.space_ops import rotate_vector
# from manimlib.utils.space_ops import rotation_matrix


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


def fractalify(vmobject, order=3, *args, **kwargs):
    """
    分型函数的调用函数
    使用一个 for 循环, 对图形对象进行 order 次的分形化迭代。
    每次迭代都会调用 fractalification_iteration 函数, 该函数的作用是将图形对象的每个子对象
    (submobject)替换为一个由多个子对象组成的新图形对象。
    当然也可以传入其它的参数来对vmobject进行处理?
    """
    for x in range(order):
        fractalification_iteration(vmobject)
    return vmobject


def fractalification_iteration(
    vmobject, dimension=1.05, num_inserted_anchors_range=list(range(1, 4))
):
    """
    这是一个用于生成分形图形的PYTHON函数。它的参数是:

    - vmobject: 一个表示矢量图形对象的变量, 可以是点、线、曲线、多边形等。
    - dimension: 一个表示分形维度的浮点数, 用于控制分形图形的复杂度和粗糙度。维度越大, 分形越复杂和粗糙。默认值是1.05。
    - num_inserted_anchors_range: 一个表示在每个锚点之间插入多少个新锚点的整数列表, 用于控制分形图形的细节和变化。列表中的每个元素都是一个可能的插入数量, 函数会随机选择一个。默认值是[1, 2, 3]。

    函数的主要逻辑是:

    - 遍历vmobject的所有锚点, 对于相邻的两个锚点p1和p2, 根据num_inserted_anchors_range随机选择一个插入数量num_inserts, 然后在p1和p2之间均匀地插入num_inserts个新锚点。
    - 对于每个新锚点, 根据dimension和p1和p2之间的距离计算一个偏移量offset_len, 然后沿着p1和p2连线的垂直方向偏移offset_len的距离, 交替地向上或向下偏移。
    - 将所有的原始锚点和新锚点连接起来, 形成一个新的vmobject。
    - 对于vmobject的所有子对象, 递归地调用函数自身, 生成更多的分形细节。

    函数的返回值是一个经过分形化处理的vmobject。
    """
    num_points = vmobject.get_num_points()  # 获取接收到的图像的所有锚点
    if num_points > 0:  # 只有在锚点数目大于0的时候才会进行后续处理
        # original_anchors = vmobject.get_anchors()
        original_anchors = [
            vmobject.point_from_proportion(x)
            for x in np.linspace(0, 1 - 1.0 / num_points, num_points)
        ]  # 这句话是得到这一个路径上所有的点的意思，因point_from_proportion里面的参数只能从0到1 取值
        # 如果为0就是在路径的起点，如果是1，那就是在终点，如果是0.5那就是在路径的中点
        new_anchors = []
        for (
            p1,
            p2,
        ) in zip(original_anchors, original_anchors[1:]):
            num_inserts = random.choice(num_inserted_anchors_range)
            inserted_points = [
                interpolate(p1, p2, alpha)
                for alpha in np.linspace(0, 1, num_inserts + 2)[1:-1]
            ]
            mass_scaling_factor = 1.0 / (num_inserts + 1)
            length_scaling_factor = mass_scaling_factor ** (1.0 / dimension)
            target_length = np.linalg.norm(p1 - p2) * length_scaling_factor
            curr_length = np.linalg.norm(p1 - p2) * mass_scaling_factor
            # offset^2 + curr_length^2 = target_length^2
            offset_len = np.sqrt(target_length**2 - curr_length**2)
            unit_vect = (p1 - p2) / np.linalg.norm(p1 - p2)
            offset_unit_vect = rotate_vector(unit_vect, np.pi / 2)
            inserted_points = [
                point + u * offset_len * offset_unit_vect
                for u, point in zip(it.cycle([-1, 1]), inserted_points)
            ]
            new_anchors += [p1] + inserted_points
        new_anchors.append(original_anchors[-1])
        vmobject.set_points_as_corners(new_anchors)
    vmobject.add(
        *[
            fractalification_iteration(submob, dimension, num_inserted_anchors_range)
            for submob in vmobject.submobjects
        ]
    )
    return vmobject


def selfsimilarfractal_iteration(vgroup, num, classname):
    """
    这是一个针对SelfSimilarFractal的子类的分型方法
    它的作用是将传入的原先的图像vgroup复制num份，然后将这些份分别按照
    传入的classname().arrange_subparts方法来进行位置排列
    并且传出一个VGroup类型的变量，这个变量里面包含的图形已经被处理好了
    其中的子部份已经被排列到正确的位置上。如果相对某个分型进行进一步的迭代的话，反复调用这个函数就好。
    为什么不把这个方法写在3B1B的SelfSimilarFractal父类里面？因为我是懒狗不想改别人的代码
    """
    # 再次强调classname仅限于
    # SelfSimilarFractal的那些描述分型的子类，因为只有他们有arrange_subparts方法
    # 父类SelfSimilarFractal的arrange_subparts是一个抽象方法
    # 将原先的vgroup复制num份，装入列表中
    var = [vgroup.copy() for i in range(num)]
    # 把迭代的vgroup的位置放到位
    classname().arrange_subparts(*var)  # 这时候就把子图像的的位置放到位了
    # 不需要设置任何的原函数参数返回
    # 把原先的列表转化为VGroup，不然的话不好调用后面的move_to和scale等方法
    vgroup = VGroup()
    vgroup.add(*var)
    # 把VGroup移动到原点
    vgroup.move_to(ORIGIN)
    # 返回VGroup对象
    return vgroup


class SelfSimilarFractal(VMobject):
    """
    这段代码是一个用于定义自相似分形图形的类，它继承了VMobject类，这是一个表示矢量图形对象的基类。它的属性和方法是：

    - order: 一个表示分形的阶数的整数，用于控制分形的递归深度。阶数越高，分形越复杂。默认值是5。
    - num_subparts: 一个表示每个分形部分包含多少个子部分的整数，用于控制分形的分支数量。默认值是3。
    - height: 一个表示分形图形的高度的浮点数，用于控制分形的大小。默认值是4。
    - colors: 一个表示分形图形的颜色渐变的列表，用于控制分形的外观。默认值是[RED, WHITE]。
    - stroke_width: 一个表示分形图形的描边宽度的浮点数，用于控制分形的边缘粗细。默认值是1。注意，这不是正确的方式来设置描边宽度，应该使用set_stroke()方法。
    - fill_opacity: 一个表示分形图形的填充不透明度的浮点数，用于控制分形的透明度。默认值是1。注意，这不是正确的方式来设置填充不透明度，应该使用set_fill()方法。

    - init_colors(self): 一个用于初始化分形图形的颜色的方法，它调用了VMobject类的init_colors()方法，并使用set_color_by_gradient()方法来设置颜色渐变。
    - init_points(self): 一个用于初始化分形图形的点集的方法，它调用了get_order_n_self()方法来获取指定阶数的分形图形，并将其设置为自身的子对象。如果阶数为0，则直接设置为单个对象；否则，设置为多个对象。
    - get_order_n_self(self, order): 一个用于获取指定阶数的分形图形的方法，它使用递归算法来生成分形图形。如果阶数为0，则返回种子图形；否则，复制上一阶数的分形图形num_subparts次，并使用arrange_subparts()方法来排列子部分，然后将它们组合成一个VGroup对象。最后，将结果设置为指定高度并居中。

    这个类可以用于创建各种自相似分形图形，例如科赫曲线、谢尔宾斯基三角、曼德勃罗集等。

    """

    order = 5  # 一个表示分形的阶数的整数，用于控制分形的递归深度。阶数越高，分形越复杂。默认值是5。
    num_subparts = 3  # 一个表示每个分形部分包含多少个子部分的整数，用于控制分形的分支数量。默认值是3。
    height = 4  # 一个表示分形图形的高度的浮点数，用于控制分形的大小。默认值是4。
    colors = [RED, WHITE]  # 一个表示分形图形的颜色渐变的列表，用于控制分形的外观。默认值是[RED, WHITE]
    stroke_width = 1  # 一个表示分形图形的描边宽度的浮点数，用于控制分形的边缘粗细。默认值是1。
    # 注意，这不是正确的方式来设置描边宽度，应该使用set_stroke()方法。Not the right way to assign stroke width
    fill_opacity = 1  # 一个表示分形图形的填充不透明度的浮点数，用于控制分形的透明度。默认值是1。
    # 注意，这不是正确的方式来设置填充不透明度，应该使用set_fill()方法。Not the right way to assign fill opacity

    def init_colors(self):
        VMobject.init_colors(self)
        self.set_color_by_gradient(*self.colors)

    def init_points(self):
        order_n_self = self.get_order_n_self(self.order)
        if self.order == 0:
            self.add(*[order_n_self])
        else:
            self.add(order_n_self.submobjects)
        return self

    def get_order_n_self(self, order):
        if order == 0:
            result = self.get_seed_shape()
        else:
            lower_order = self.get_order_n_self(order - 1)
            subparts = [lower_order.copy() for x in range(self.num_subparts)]
            self.arrange_subparts(*subparts)
            result = VGroup(*subparts)

        result.set_height(self.height)
        result.center()
        return result

    def get_seed_shape(self):
        raise Exception("Not implemented")

    def arrange_subparts(self, *subparts):
        raise Exception("Not implemented")


class Sierpinski(SelfSimilarFractal):
    def get_seed_shape(self):
        return Polygon(
            RIGHT,
            np.sqrt(3) * UP,
            LEFT,
        )

    def arrange_subparts(self, *subparts):
        tri1, tri2, tri3 = subparts
        tri1.move_to(tri2.get_corner(DOWN + LEFT), UP)
        tri3.move_to(tri2.get_corner(DOWN + RIGHT), UP)


class TestSierpinski(Scene):
    """
    这是一份用来检测上面的Sierpinski类有没有出问题的一个Scene
    结果显示非常正常，当然也顺便爽渲染了一把谢尔宾斯基三角形
    """

    def construct(self):
        sierpinskistage = VGroup()
        sierpinskistage.add(Sierpinski().get_seed_shape())  # 得到种子图像
        sierpinskistage.set_opacity(0.5).set_color(
            [RED, YELLOW, BLUE]
        )  # 顺便调整种子图像的透明度与颜色

        # 调用sierpinsk_ita开始迭代
        # 不写一整个循环的主要原因是要手动调整三角形的大小，，，
        # 如果一直只缩放0.75倍的话，后面迭代的话就会超出屏幕
        for i in range(3):
            self.play(
                Transform(
                    sierpinskistage,
                    selfsimilarfractal_iteration(sierpinskistage, 3, Sierpinski).scale(
                        0.75
                    ),
                )
            )
        for i in range(3):
            self.play(
                Transform(
                    sierpinskistage,
                    selfsimilarfractal_iteration(sierpinskistage, 3, Sierpinski).scale(
                        0.55
                    ),
                )
            )


class RenderCover(Scene):
    """
    这个是用来渲染封面的
    """

    def construct(self):
        text = Text("Sierpinski Carpet", weight=BOLD, font_size=40)
        text.move_to(DOWN * 3.75)

        var = Sierpinski().get_seed_shape()
        var.set_opacity(0.55).set_color([RED, YELLOW, BLUE])
        sierpinski = VGroup().add(var)

        for i in range(3):
            sierpinski = TestSierpinski.sierpinsk_ita(self, sierpinski).scale(0.8)
        sierpinski.move_to(UP * 0.15)

        self.add(sierpinski, text)


class DiamondFractal(SelfSimilarFractal):
    num_subparts = 4
    height = 4
    colors = [GREEN_E, YELLOW]

    def get_seed_shape(self):
        return RegularPolygon(n=4)

    def arrange_subparts(self, *subparts):
        # VGroup(*subparts).rotate(np.pi/4)
        for part, vect in zip(subparts, compass_directions(start_vect=UP + RIGHT)):
            part.next_to(ORIGIN, vect, buff=0)
        VGroup(*subparts).rotate(np.pi / 4, about_point=ORIGIN)


class TestDiamondFractal(Scene):
    """
    没错，这个测试代码是按照上面的谢尔宾斯基三角形的测试代码改过来的
    渲染时间较长，稍安勿躁
    """

    def construct(self):
        diamondstage = VGroup()
        diamondstage.add(DiamondFractal().get_seed_shape())  # 得到种子图像
        diamondstage.set_opacity(0.5).set_color([RED, YELLOW, BLUE])  # 顺便调整种子图像的透明度与颜色

        # 调用diamond_ita开始迭代
        # 不写一整个循环的主要原因是要手动调整四边形的大小，，，
        # 如果一直只缩放0.65倍的话，后面迭代的话就会超出屏幕
        for i in range(3):
            self.play(
                Transform(
                    diamondstage,
                    selfsimilarfractal_iteration(diamondstage, 4, DiamondFractal).scale(
                        0.65
                    ),
                )
            )
        for i in range(3):
            self.play(
                Transform(
                    diamondstage,
                    selfsimilarfractal_iteration(diamondstage, 4, DiamondFractal).scale(
                        0.45
                    ),
                )
            )


class PentagonalFractal(SelfSimilarFractal):
    num_subparts = 5
    colors = [MAROON_B, YELLOW, RED]
    height = 6

    def get_seed_shape(self):
        return RegularPolygon(n=5, start_angle=np.pi / 2)

    def arrange_subparts(self, *subparts):
        for x, part in enumerate(subparts):
            part.shift(0.95 * part.get_height() * UP)
            part.rotate(2 * np.pi * x / 5, about_point=ORIGIN)


class TestPentagonal(Scene):
    """
    测试五边形分型的可行性，，代码还是照抄上面的
    """

    def construct(self):
        penstage = VGroup()
        var = PentagonalFractal.get_seed_shape(self)
        penstage.add(var)

        for i in range(3):
            self.play(
                Transform(
                    penstage,
                    selfsimilarfractal_iteration(penstage, 5, PentagonalFractal).scale(
                        0.575
                    ),
                )
            )


class PentagonalPiCreatureFractal(PentagonalFractal):
    """
    这是一段使用PICreature作为分型的基本单元而出现的代码，因为manimce版本里面并没有PICreature
    所以说我也不知道该怎么办，但是既然继承自上面的PentagonalFractal五边形分型，想必也一定很聪明吧（智将）
    """

    def init_colors(self):
        SelfSimilarFractal.init_colors(self)
        internal_pis = [pi for pi in self.get_family() if isinstance(pi, PiCreature)]
        colors = color_gradient(self.colors, len(internal_pis))
        for pi, color in zip(internal_pis, colors):
            pi.init_colors()
            pi.body.set_stroke(color, width=0.5)
            pi.set_color(color)

    def get_seed_shape(self):
        return Randolph(mode="shruggie")

    def arrange_subparts(self, *subparts):
        for part in subparts:
            part.rotate(2 * np.pi / 5, about_point=ORIGIN)
        PentagonalFractal.arrange_subparts(self, *subparts)


class PiCreatureFractal(VMobject):
    order = 7
    scale_val = 2.5
    start_mode = "hooray"
    height = 6
    colors = [
        BLUE_D,
        BLUE_B,
        MAROON_B,
        MAROON_D,
        GREY,
        YELLOW,
        RED,
        GREY_BROWN,
        RED,
        RED_E,
    ]
    random_seed = 0
    stroke_width = 0

    def init_colors(self):
        VMobject.init_colors(self)
        internal_pis = [pi for pi in self.get_family() if isinstance(pi, PiCreature)]
        random.seed(self.random_seed)
        for pi in reversed(internal_pis):
            color = random.choice(self.colors)
            pi.set_color(color)
            pi.set_stroke(color, width=0)

    def init_points(self):
        random.seed(self.random_seed)
        modes = get_all_pi_creature_modes()
        seed = PiCreature(mode=self.start_mode)
        seed.set_height(self.height)
        seed.to_edge(DOWN)
        creatures = [seed]
        self.add(VGroup(seed))
        for x in range(self.order):
            new_creatures = []
            for creature in creatures:
                for eye, vect in zip(creature.eyes, [LEFT, RIGHT]):
                    new_creature = PiCreature(mode=random.choice(modes))
                    new_creature.set_height(self.scale_val * eye.get_height())
                    new_creature.next_to(eye, vect, buff=0, aligned_edge=DOWN)
                    new_creatures.append(new_creature)
                creature.look_at(random.choice(new_creatures))
            self.add_to_back(VGroup(*new_creatures))
            creatures = new_creatures

    # def init_colors(self):
    #     VMobject.init_colors(self)
    #     self.set_color_by_gradient(*self.colors)


class WonkyHexagonFractal(SelfSimilarFractal):
    num_subparts = 7

    def get_seed_shape(self):
        return RegularPolygon(n=6)

    def arrange_subparts(self, *subparts):
        for i, piece in enumerate(subparts):
            piece.rotate(i * np.pi / 12, about_point=ORIGIN)
        p1, p2, p3, p4, p5, p6, p7 = subparts
        center_row = VGroup(p1, p4, p7)
        center_row.arrange(RIGHT, buff=0)
        for p in p2, p3, p5, p6:
            p.set_width(p1.get_width())
        p2.move_to(p1.get_top(), DOWN + LEFT)
        p3.move_to(p1.get_bottom(), UP + LEFT)
        p5.move_to(p4.get_top(), DOWN + LEFT)
        p6.move_to(p4.get_bottom(), UP + LEFT)


class TestWonkyHexagon(Scene):
    def construct(self):
        penstage = VGroup()
        var = WonkyHexagonFractal.get_seed_shape(self)
        penstage.add(var)

        for i in range(3):
            self.play(
                Transform(
                    penstage,
                    selfsimilarfractal_iteration(
                        penstage, 7, WonkyHexagonFractal
                    ).scale(0.5),
                )
            )


class CircularFractal(SelfSimilarFractal):
    num_subparts = 3
    colors = [GREEN, BLUE, GREY]

    def get_seed_shape(self):
        return Circle()

    def arrange_subparts(self, *subparts):
        if not hasattr(self, "been_here"):
            self.num_subparts = 3 + self.order
            self.been_here = True
        for i, part in enumerate(subparts):
            theta = np.pi / self.num_subparts
            part.next_to(ORIGIN, UP, buff=self.height / (2 * np.tan(theta)))
            part.rotate(i * 2 * np.pi / self.num_subparts, about_point=ORIGIN)
        self.num_subparts -= 1


class TestCircular(Scene):
    def construct(self):
        penstage = VGroup()
        var = CircularFractal.get_seed_shape(self)
        penstage.add(var)

        for i in range(6):
            self.play(
                Transform(
                    penstage,
                    selfsimilarfractal_iteration(penstage, 3, CircularFractal).scale(
                        0.5
                    ),
                )
            )


######## Space filling curves ############


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


class HilbertCurve3D(SelfSimilarSpaceFillingCurve):
    offsets = [
        RIGHT + DOWN + IN,
        LEFT + DOWN + IN,
        LEFT + DOWN + OUT,
        RIGHT + DOWN + OUT,
        RIGHT + UP + OUT,
        LEFT + UP + OUT,
        LEFT + UP + IN,
        RIGHT + UP + IN,
    ]
    offset_to_rotation_axis_and_angle = {
        str(RIGHT + DOWN + IN): (LEFT + UP + OUT, 2 * np.pi / 3),
        str(LEFT + DOWN + IN): (RIGHT + DOWN + IN, 2 * np.pi / 3),
        str(LEFT + DOWN + OUT): (RIGHT + DOWN + IN, 2 * np.pi / 3),
        str(RIGHT + DOWN + OUT): (UP, np.pi),
        str(RIGHT + UP + OUT): (UP, np.pi),
        str(LEFT + UP + OUT): (LEFT + DOWN + OUT, 2 * np.pi / 3),
        str(LEFT + UP + IN): (LEFT + DOWN + OUT, 2 * np.pi / 3),
        str(RIGHT + UP + IN): (RIGHT + UP + IN, 2 * np.pi / 3),
    }
    # Rewrote transform method to include the rotation angle

    def transform(self, points, offset):
        copy = np.array(points)
        copy = rotate(
            copy,
            axis=self.offset_to_rotation_axis_and_angle[str(offset)][0],
            angle=self.offset_to_rotation_axis_and_angle[str(offset)][1],
        )
        copy /= (self.scale_factor,)
        copy += offset * self.radius * self.radius_scale_factor
        return copy


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


class TriangleFillingCurve(SelfSimilarSpaceFillingCurve):
    colors = [MAROON, YELLOW]
    offsets = [
        LEFT / 4.0 + DOWN / 6.0,
        ORIGIN,
        RIGHT / 4.0 + DOWN / 6.0,
        UP / 3.0,
    ]
    offset_to_rotation_axis = {
        str(ORIGIN): RIGHT,
        str(UP / 3.0): UP,
    }
    scale_factor = 2
    radius_scale_factor = 1.5


class HexagonFillingCurve(SelfSimilarSpaceFillingCurve):
    start_color = WHITE
    end_color = BLUE_D
    axis_offset_pairs = [
        (None, 1.5 * DOWN + 0.5 * np.sqrt(3) * LEFT),
        (UP + np.sqrt(3) * RIGHT, 1.5 * DOWN + 0.5 * np.sqrt(3) * RIGHT),
        (np.sqrt(3) * UP + RIGHT, ORIGIN),
        ((UP, RIGHT), np.sqrt(3) * LEFT),
        (None, 1.5 * UP + 0.5 * np.sqrt(3) * LEFT),
        (None, 1.5 * UP + 0.5 * np.sqrt(3) * RIGHT),
        (RIGHT, np.sqrt(3) * RIGHT),
    ]
    scale_factor = 3
    radius_scale_factor = 2 / (3 * np.sqrt(3))

    def refine_into_subparts(self, points):
        return SelfSimilarSpaceFillingCurve.refine_into_subparts(
            self, rotate(points, np.pi / 6, IN)
        )


class UtahFillingCurve(SelfSimilarSpaceFillingCurve):
    colors = [WHITE, BLUE_D]
    axis_offset_pairs = []
    scale_factor = 3
    radius_scale_factor = 2 / (3 * np.sqrt(3))


class FlowSnake(LindenmayerCurve):
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


class SierpinskiCurve(LindenmayerCurve):
    colors = [RED, WHITE]
    axiom = "B"
    rule = {
        "A": "+B-A-B+",
        "B": "-A+B+A-",
    }
    radius = 6  # TODO, this is innaccurate
    scale_factor = 2
    start_step = RIGHT
    angle = -np.pi / 3


class KochSnowFlake(LindenmayerCurve):
    colors = [BLUE_D, WHITE, BLUE_D]
    axiom = "A--A--A--"
    rule = {"A": "A+A--A+A"}
    radius = 4
    scale_factor = 3
    start_step = RIGHT
    angle = np.pi / 3
    order_to_stroke_width_map = {
        3: 3,
        5: 2,
        6: 1,
    }

    def __init__(self, **kwargs):
        digest_config(self, kwargs)
        self.scale_factor = 2 * (1 + np.cos(self.angle))
        LindenmayerCurve.__init__(self, **kwargs)


class KochCurve(KochSnowFlake):
    axiom = "A--"


class QuadraticKoch(LindenmayerCurve):
    colors = [YELLOW, WHITE, MAROON_B]
    axiom = "A"
    rule = {"A": "A+A-A-AA+A+A-A"}
    radius = 4
    scale_factor = 4
    start_step = RIGHT
    angle = np.pi / 2


class QuadraticKochIsland(QuadraticKoch):
    axiom = "A+A+A+A"


class StellarCurve(LindenmayerCurve):
    start_color = RED
    end_color = BLUE_E
    rule = {
        "A": "+B-A-B+A-B+",
        "B": "-A+B+A-B+A-",
    }
    scale_factor = 3
    angle = 2 * np.pi / 5


class SnakeCurve(FractalCurve):
    start_color = BLUE
    end_color = YELLOW

    def get_anchor_points(self):
        result = []
        resolution = 2**self.order
        step = 2.0 * self.radius / resolution
        lower_left = (
            ORIGIN + LEFT * (self.radius - step / 2) + DOWN * (self.radius - step / 2)
        )

        for y in range(resolution):
            x_range = list(range(resolution))
            if y % 2 == 0:
                x_range.reverse()
            for x in x_range:
                result.append(lower_left + x * step * RIGHT + y * step * UP)
        return result
