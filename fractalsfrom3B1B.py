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

    # 定义一个实例方法，用于初始化点集，并且进行连线
    # 这是VMobject类的一个重写方法
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


# 定义一个LindenmayerCurve类，继承自FractalCurve类
class LindenmayerCurve(FractalCurve):
    """
        这段代码中，LindenmayerCurve类有以下几个属性和方法：

    axiom：一个字符串，表示L系统的初始符号，这里使用了"A"。
    rule：一个字典，表示L系统的产生规则，这里使用了一个空字典。
    scale_factor：一个整数，表示每次迭代时缩放的比例，这里使用了2。
    radius：一个整数，表示绘制L系统时使用的半径，这里使用了3。
    start_step：一个常量，表示绘制L系统时开始的方向，这里使用了RIGHT，表示向右。
    angle：一个浮点数，表示每次转弯时的角度，这里使用了np.pi / 2，表示90度。
    expand_command_string：一个方法，用于将一个命令字符串按照产生规则进行扩展，返回一个新的命令字符串。它的参数是command，表示要扩展的命令字符串。它的返回值是result，表示扩展后的命令字符串。它的工作流程是：
    首先，它创建一个空字符串result。
    然后，它遍历command中的每个字母letter。
    接着，它检查letter是否在rule中有对应的值，如果有，就将值添加到result中；如果没有，就将letter本身添加到result中。
    最后，它返回result。
    get_command_string：一个方法，用于根据迭代次数（order）生成最终的命令字符串。它没有参数。它的返回值是result，表示最终的命令字符串。它的工作流程是：
    首先，它将axiom赋值给result。
    然后，它根据order进行循环迭代。
    接着，它调用expand_command_string方法对result进行扩展，并将返回值重新赋值给result。
    最后，它返回result。
    get_anchor_points：一个方法，用于根据命令字符串生成锚点（anchor points），锚点是用于绘制曲线的关键点。它没有参数。它的返回值是一个NumPy数组（np.array），表示锚点的坐标。它的工作流程是：
    首先，它根据radius和start_step计算出初始的步长（step），并根据scale_factor和order对其进行缩放。步长是每次移动时沿着方向移动的距离。
    然后，它创建一个零向量（np.zeros(3)），表示当前的位置（curr）。
    接着，它创建一个列表（result），并将curr添加到其中。列表中存储了所有锚点的位置。
    然后，它调用get_command_string方法获取最终的命令字符串，并遍历其中的每个字母letter。
    接着，它根据letter进行判断：
    如果letter是"+"，就将step向左旋转angle度（rotate(step, self.angle)）；
    如果letter是"-"，就将step向右旋转angle度（rotate(step, -self.angle)）；
    否则，就将curr加上step，并将新的curr添加到result中。
    最后，它将result转换成NumPy数组，并减去其质心（center_of_mass(result)），使其居中对齐，并返回该数组。
    """

    # 定义L系统的初始符号为"A"
    axiom = "A"
    # 定义L系统的产生规则为空字典
    rule = {}
    # 定义每次迭代时缩放的比例为2
    scale_factor = 2
    # 定义绘制L系统时使用的半径为3
    radius = 3
    # 定义绘制L系统时开始的方向为右
    start_step = RIGHT
    # 定义每次转弯时的角度为90度
    angle = np.pi / 2

    # 定义一个方法，用于将一个命令字符串按照产生规则进行扩展，返回一个新的命令字符串
    def expand_command_string(self, command):
        # 创建一个空字符串result
        result = ""
        # 遍历command中的每个字母letter
        for letter in command:
            # 如果letter在rule中有对应的值，就将值添加到result中；如果没有，就将letter本身添加到result中
            if letter in self.rule:
                result += self.rule[letter]
            else:
                result += letter
        # 返回result
        return result

    # 定义一个方法，用于根据迭代次数（order）生成最终的命令字符串
    def get_command_string(self):
        # 将axiom赋值给result
        result = self.axiom
        # 根据order进行循环迭代
        for x in range(self.order):
            # 调用expand_command_string方法对result进行扩展，并将返回值重新赋值给result
            result = self.expand_command_string(result)
        # 返回result
        return result

    # 定义一个方法，用于根据命令字符串生成锚点（anchor points），锚点是用于绘制曲线的关键点
    def get_anchor_points(self):
        # 根据radius和start_step计算出初始的步长（step），并根据scale_factor和order对其进行缩放。步长是每次移动时沿着方向移动的距离。
        step = float(self.radius) * self.start_step
        step /= self.scale_factor**self.order
        # 创建一个零向量（np.zeros(3)），表示当前的位置（curr）
        curr = np.zeros(3)
        # 创建一个列表（result），并将curr添加到其中。列表中存储了所有锚点的位置。
        result = [curr]
        # 调用get_command_string方法获取最终的命令字符串，并遍历其中的每个字母letter
        for letter in self.get_command_string():
            # 根据letter进行判断：
            if letter == "+":
                # 如果letter是"+"，就将step向左旋转angle度（rotate(step, self.angle)）
                step = rotate(step, self.angle)
            elif letter == "-":
                # 如果letter是"-"，就将step向右旋转angle度（rotate(step, -self.angle)）
                step = rotate(step, -self.angle)
            else:
                # 否则，就将curr加上step，并将新的curr添加到result中。
                curr = curr + step
                result.append(curr)
        # 将result转换成NumPy数组，并减去其质心（center_of_mass(result)），使其居中对齐，并返回该数组。
        return np.array(result) - center_of_mass(result)


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


class TestHilbertCurve(Scene):
    def construct(self):
        num = 7
        hilbert_group = [HilbertCurve() for i in range(num)]
        # 实例化了一次之后这个实例在循环里面被调用时候应该使用copy方法复制一份
        # 不然的话循环里面重复调用同一个实例化的物体势必会出问题
        # 非要在循环里面重复调用同一个类的话应该重复实例化多次再加进去

        for i in range(num):
            hilbert_group[i].order = i + 1
            hilbert_group[i].init_points()  # 原来父类里面写了怎么连线的方法，，大意了没有闪
            # 下一次一定会好好看父类的！
            hilbert_group[i].init_colors()
            hilbert_group[i].set_stroke(width=4)  # 设置曲线宽度为固定值

        self.play(Create(hilbert_group[0]))
        self.wait()

        for i in range(len(hilbert_group) - 1):
            self.play(ReplacementTransform(hilbert_group[i], hilbert_group[i + 1]))
        self.wait()


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


class TestPeanoCurve(Scene):
    def construct(self):
        num = 4
        peano_group = [PeanoCurve() for i in range(num)]
        # 实例化了一次之后这个实例在循环里面被调用时候应该使用copy方法复制一份
        # 不然的话循环里面重复调用同一个实例化的物体势必会出问题
        # 非要在循环里面重复调用同一个类的话应该重复实例化多次再加进去

        for i in range(num):
            peano_group[i].order = i + 1
            peano_group[i].init_points()  # 原来父类里面写了怎么连线的方法，，大意了没有闪
            # 下一次一定会好好看父类的！
            peano_group[i].init_colors()
            peano_group[i].set_stroke(width=4)  # 设置曲线宽度为固定值

        self.play(Create(peano_group[0]))
        self.wait()

        for i in range(len(peano_group) - 1):
            self.play(ReplacementTransform(peano_group[i], peano_group[i + 1]))
        self.wait()


class RenderCover2(Scene):
    # 探究不同的set_points方法对结果的影响
    def construct(self):
        # #古典派自己撸的渲染方案，现在都不这么写了，因为我发现真正的连线的方案被放在父类里面了（悲
        # peano_curve = PeanoCurve()
        # peano_curve.order = 2
        # points = peano_curve.get_anchor_points()
        # peano_curve.set_stroke(width=4)  # 设置线段宽度

        # pe1 = peano_curve.copy()
        # pe1.colors = [RED, GREEN]

        # peano_curve.set_points(points)
        # pe1.set_points_as_corners(points)

        # dots = VGroup()
        # for var in points:
        #     dots.add(Dot(radius=0.01).move_to(var))
        # self.add(dots)
        # self.play(Create(pe1), run_time=2, rate_func=lambda t: t)
        # self.wait()
        # self.play(
        #     Create(peano_curve),
        #     run_time=2,
        # )

        peano_curve = PeanoCurve()
        peano_curve.set_stroke(width=4)  # 设置线段宽度
        peano_curve.order = 2  # 这个是父类的一个特性，阶数，在子类得到继承，现在修改一下


        pe1 = peano_curve.copy()  # pe1曲线是跟原先的曲线相同的阶数与线段宽度，但是懒得再写阶数直接开始复制算了
        peano_curve.init_points()
        peano_curve.init_colors()  # 这个是父类的两个方法，调用了这两个方法就相当于把线连接好并且把颜色设置好了
        # ，就不需要自己调用VMobject里面的get_anchor_points方法来自己撸了
        peano_curve.scale(1.45)

        points = (
            peano_curve.get_anchor_points()
        )  # 获取曲线的锚点坐标，后续会用另一种VMobject的set_points方法来连线构成pe1
        # 方便与本体的连线方法作比较

        pe1.set_points(points)  # 用VMobject里面的set_points方法来连线

        dots = VGroup()  # 创建一个vgroup，用来储存用原先的锚点坐标绘制的点对象
        for var in points:
            dots.add(Dot(radius=0.01).move_to(var*1.45))

        self.add(dots)  # 在平面上加上基于这些锚点坐标的点

        self.play(
            Create(peano_curve), run_time=2, rate_func=lambda t: t
        )  # 在平面上绘制pe1，用线性速度绘制
        self.wait()
        self.play(
            Create(pe1.scale(1.45)),
            run_time=2,
        )  # 绘制原先连线方式的peano曲线，绘制的速度为smooth，与线性速度做对比。


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


class TestTriangleFillingCurve(Scene):
    def construct(self):
        num = 5
        triangle_group = [TriangleFillingCurve() for i in range(num)]
        # 实例化了一次之后这个实例在循环里面被调用时候应该使用copy方法复制一份
        # 不然的话循环里面重复调用同一个实例化的物体势必会出问题
        # 非要在循环里面重复调用同一个类的话应该重复实例化多次再加进去

        for i in range(num):
            triangle_group[i].order = i + 1
            triangle_group[i].init_points()  # 原来父类里面写了怎么连线的方法，，大意了没有闪
            # 下一次一定会好好看父类的！
            triangle_group[i].init_colors()
            triangle_group[i].set_stroke(width=4)  # 设置曲线宽度为固定值

        self.play(Create(triangle_group[0]))
        self.wait()

        for i in range(len(triangle_group) - 1):
            self.play(ReplacementTransform(triangle_group[i], triangle_group[i + 1]))
        self.wait()


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


class TestFlowSnake(Scene):
    def construct(self):
        num = 5
        flowsnake_group = [FlowSnake() for i in range(num)]
        # 实例化了一次之后这个实例在循环里面被调用时候应该使用copy方法复制一份
        # 不然的话循环里面重复调用同一个实例化的物体势必会出问题
        # 非要在循环里面重复调用同一个类的话应该重复实例化多次再加进去

        for i in range(num):
            flowsnake_group[i].order = i + 2
            flowsnake_group[i].init_points()  # 原来父类里面写了怎么连线的方法，，大意了没有闪
            # 下一次一定会好好看父类的！
            flowsnake_group[i].init_colors()
            flowsnake_group[i].set_stroke(width=4)  # 设置曲线宽度为固定值

        self.play(Create(flowsnake_group[0]))
        self.wait()

        for i in range(len(flowsnake_group) - 1):
            self.play(ReplacementTransform(flowsnake_group[i], flowsnake_group[i + 1]))
        self.wait()


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


class TestSierpinskiCurve(Scene):
    def construct(self):
        num = 6
        sierpinski_group = [SierpinskiCurve() for i in range(num)]
        # 实例化了一次之后这个实例在循环里面被调用时候应该使用copy方法复制一份
        # 不然的话循环里面重复调用同一个实例化的物体势必会出问题
        # 非要在循环里面重复调用同一个类的话应该重复实例化多次再加进去

        for i in range(num):
            sierpinski_group[i].order = i + 2
            sierpinski_group[i].init_points()  # 原来父类里面写了怎么连线的方法，，大意了没有闪
            # 下一次一定会好好看父类的！
            sierpinski_group[i].init_colors()
            sierpinski_group[i].set_stroke(width=4)  # 设置曲线宽度为固定值

        self.play(Create(sierpinski_group[0]))
        self.wait()

        for i in range(len(sierpinski_group) - 1):
            self.play(
                ReplacementTransform(sierpinski_group[i], sierpinski_group[i + 1])
            )
        self.wait()


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

    # def __init__(self, **kwargs):
    #     # 将属性作为关键字参数传递给父类的构造函数
    #     LindenmayerCurve.__init__(
    #         self,
    #         colors=self.colors,
    #         axiom=self.axiom,
    #         rule=self.rule,
    #         radius=self.radius,
    #         scale_factor=self.scale_factor,
    #         start_step=self.start_step,
    #         angle=self.angle,
    #         order_to_stroke_width_map=self.order_to_stroke_width_map,
    #         **kwargs
    #     )
    #     # 删除digest_config函数的调用
    #     # digest_config(self, kwargs)
    #     # 将scale_factor属性放在父类构造函数之后
    #     self.scale_factor = 2 * (1 + np.cos(self.angle))
    #     self.set_points_as_corners([*self.generate_points(), self.points[0]])


class TestKochSnowFlake(Scene):
    def construct(self):
        num = 3
        kochsnow_group = [KochSnowFlake() for i in range(num)]
        # 实例化了一次之后这个实例在循环里面被调用时候应该使用copy方法复制一份
        # 不然的话循环里面重复调用同一个实例化的物体势必会出问题
        # 非要在循环里面重复调用同一个类的话应该重复实例化多次再加进去

        for i in range(num):
            kochsnow_group[i].order = i + 2
            kochsnow_group[i].init_points()  # 原来父类里面写了怎么连线的方法，，大意了没有闪
            # 下一次一定会好好看父类的！
            kochsnow_group[i].init_colors()
            # kochsnow_group[i].set_stroke(width=4)  # 设置曲线宽度为固定值

        self.play(Create(kochsnow_group[0]))
        self.wait()

        for i in range(len(kochsnow_group) - 1):
            self.play(ReplacementTransform(kochsnow_group[i], kochsnow_group[i + 1]))
        self.wait()


class KochCurve(KochSnowFlake):
    axiom = "A--"


class TestKochCurve(Scene):
    def construct(self):
        num = 3
        kochcurve_group = [KochCurve() for i in range(num)]
        # 实例化了一次之后这个实例在循环里面被调用时候应该使用copy方法复制一份
        # 不然的话循环里面重复调用同一个实例化的物体势必会出问题
        # 非要在循环里面重复调用同一个类的话应该重复实例化多次再加进去

        for i in range(num):
            kochcurve_group[i].order = i + 2
            kochcurve_group[i].init_points()  # 原来父类里面写了怎么连线的方法，，大意了没有闪
            # 下一次一定会好好看父类的！
            kochcurve_group[i].init_colors()
            # kochcurve_group[i].set_stroke(width=4)  # 设置曲线宽度为固定值

        self.play(Create(kochcurve_group[0]))
        self.wait()

        for i in range(len(kochcurve_group) - 1):
            self.play(ReplacementTransform(kochcurve_group[i], kochcurve_group[i + 1]))
        self.wait()


class QuadraticKoch(LindenmayerCurve):
    colors = [YELLOW, WHITE, MAROON_B]
    axiom = "A"
    rule = {"A": "A+A-A-AA+A+A-A"}
    radius = 4
    scale_factor = 4
    start_step = RIGHT
    angle = np.pi / 2


class TestQuadraticKoch(Scene):
    def construct(self):
        num = 4
        quadratic_group = [QuadraticKoch() for i in range(num)]
        # 实例化了一次之后这个实例在循环里面被调用时候应该使用copy方法复制一份
        # 不然的话循环里面重复调用同一个实例化的物体势必会出问题
        # 非要在循环里面重复调用同一个类的话应该重复实例化多次再加进去

        for i in range(num):
            quadratic_group[i].order = i + 2
            quadratic_group[i].init_points()  # 原来父类里面写了怎么连线的方法，，大意了没有闪
            # 下一次一定会好好看父类的！
            quadratic_group[i].init_colors()
            # quadratic_group[i].set_stroke(width=4)  # 设置曲线宽度为固定值
            quadratic_group[i].scale(3)  # 进行缩放

        self.play(Create(quadratic_group[0]))
        self.wait()

        for i in range(len(quadratic_group) - 1):
            self.play(ReplacementTransform(quadratic_group[i], quadratic_group[i + 1]))
        self.wait()


class QuadraticKochIsland(QuadraticKoch):
    axiom = "A+A+A+A"


class TestQuadraticKochIsland(Scene):
    def construct(self):
        num = 3
        quadratickocgisland_group = [QuadraticKochIsland() for i in range(num)]
        # 实例化了一次之后这个实例在循环里面被调用时候应该使用copy方法复制一份
        # 不然的话循环里面重复调用同一个实例化的物体势必会出问题
        # 非要在循环里面重复调用同一个类的话应该重复实例化多次再加进去

        for i in range(num):
            quadratickocgisland_group[i].order = i + 2
            quadratickocgisland_group[i].init_points()  # 原来父类里面写了怎么连线的方法，，大意了没有闪
            # 下一次一定会好好看父类的！
            quadratickocgisland_group[i].init_colors()
            # quadratickocgisland_group[i].set_stroke(width=4)  # 设置曲线宽度为固定值
            # quadratickocgisland_group[i].scale(3)#进行缩放

        self.play(Create(quadratickocgisland_group[0]))
        self.wait()

        for i in range(len(quadratickocgisland_group) - 1):
            self.play(
                ReplacementTransform(
                    quadratickocgisland_group[i], quadratickocgisland_group[i + 1]
                )
            )
        self.wait()


class StellarCurve(LindenmayerCurve):
    start_color = RED
    end_color = BLUE_E
    rule = {
        "A": "+B-A-B+A-B+",
        "B": "-A+B+A-B+A-",
    }
    scale_factor = 3
    angle = 2 * np.pi / 5


class TestStellarCurve(Scene):
    def construct(self):
        num = 4
        stellarcurve_group = [StellarCurve() for i in range(num)]
        # 实例化了一次之后这个实例在循环里面被调用时候应该使用copy方法复制一份
        # 不然的话循环里面重复调用同一个实例化的物体势必会出问题
        # 非要在循环里面重复调用同一个类的话应该重复实例化多次再加进去

        for i in range(num):
            stellarcurve_group[i].order = i + 2
            stellarcurve_group[i].init_points()  # 原来父类里面写了怎么连线的方法，，大意了没有闪
            # 下一次一定会好好看父类的！
            stellarcurve_group[i].init_colors()
            # stellarcurve_group[i].set_stroke(width=4)  # 设置曲线宽度为固定值
            stellarcurve_group[i].scale(2)  # 进行缩放

        self.play(Create(stellarcurve_group[0]))
        self.wait()

        for i in range(len(stellarcurve_group) - 1):
            if i < len(stellarcurve_group) - 1 - 1:
                self.play(
                    ReplacementTransform(
                        stellarcurve_group[i], stellarcurve_group[i + 1]
                    )
                )
            else:
                self.play(
                    ReplacementTransform(
                        stellarcurve_group[i], stellarcurve_group[i + 1].scale(0.9)
                    )
                )
        self.wait()


# 定义一个SnakeCurve类，继承自FractalCurve类
class SnakeCurve(FractalCurve):
    """
        SnakeCurve类是一个用于绘制蛇形曲线（snake curve）的类，蛇形曲线是一种由英国数学家布莱恩·哈罗德·普朗克特（Brian Harold Plunkett）发明的分形曲线。

    这段代码中，SnakeCurve类有以下几个属性和方法：

    start_color：一个常量，表示绘制蛇形曲线时使用的起始颜色，这里使用了BLUE，表示蓝色。
    end_color：一个常量，表示绘制蛇形曲线时使用的结束颜色，这里使用了YELLOW，表示黄色。
    get_anchor_points：一个方法，用于根据迭代次数（order）生成锚点（anchor points），锚点是用于绘制曲线的关键点。它没有参数。它的返回值是一个列表（result），表示锚点的坐标。它的工作流程是：
    首先，它根据order计算出分辨率（resolution），分辨率是每条边上的点数，它等于2的order次方。
    然后，它根据radius和resolution计算出步长（step），步长是每次移动时沿着方向移动的距离，它等于2乘以radius除以resolution。
    接着，它根据radius和step计算出左下角（lower_left）的位置，左下角是绘制蛇形曲线时开始的位置，它等于原点（ORIGIN）向左移动radius减去step除以2的距离，再向下移动radius减去step除以2的距离。
    然后，它创建一个空列表result。
    接着，它根据resolution进行垂直方向（y）的循环迭代。
    然后，它创建一个列表x_range，表示水平方向（x）上的点数范围，它等于从0到resolution-1的整数列表。
    接着，它根据y是否为偶数进行判断：
    如果y为偶数，就将x_range反转（x_range.reverse()），表示从右向左移动；
    如果y为奇数，就保持x_range不变，表示从左向右移动。
    然后，它根据x_range进行水平方向（x）的循环迭代。
    接着，它根据lower_left、x和y计算出当前位置（curr），并将其添加到result中。当前位置等于左下角加上x乘以step乘以右方向（RIGHT），再加上y乘以step乘以上方向（UP）。
    最后，它返回result。
    """

    # 定义绘制蛇形曲线时使用的起始颜色为蓝色
    start_color = BLUE
    # 定义绘制蛇形曲线时使用的结束颜色为黄色
    end_color = YELLOW

    # 定义一个方法，用于根据迭代次数（order）生成锚点（anchor points），锚点是用于绘制曲线的关键点
    def get_anchor_points(self):
        # 创建一个空列表result
        result = []
        # 根据order计算出分辨率（resolution），分辨率是每条边上的点数，它等于2的order次方
        resolution = 2**self.order
        # 根据radius和resolution计算出步长（step），步长是每次移动时沿着方向移动的距离，它等于2乘以radius除以resolution
        step = 2.0 * self.radius / resolution
        # 根据radius和step计算出左下角（lower_left）的位置，左下角是绘制蛇形曲线时开始的位置，它等于原点（ORIGIN）向左移动radius减去step除以2的距离，再向下移动radius减去step除以2的距离
        lower_left = (
            ORIGIN + LEFT * (self.radius - step / 2) + DOWN * (self.radius - step / 2)
        )

        # 根据resolution进行垂直方向（y）的循环迭代
        for y in range(resolution):
            # 创建一个列表x_range，表示水平方向（x）上的点数范围，它等于从0到resolution-1的整数列表
            x_range = list(range(resolution))
            # 根据y是否为偶数进行判断：
            if y % 2 == 0:
                # 如果y为偶数，就将x_range反转（x_range.reverse()），表示从右向左移动
                x_range.reverse()
            # 如果y为奇数，就保持x_range不变，表示从左向右移动

            # 根据x_range进行水平方向（x）的循环迭代
            for x in x_range:
                # 根据lower_left、x和y计算出当前位置（curr），并将其添加到result中。当前位置等于左下角加上x乘以step乘以右方向（RIGHT），再加上y乘以step乘以上方向（UP）
                curr = lower_left + x * step * RIGHT + y * step * UP
                result.append(curr)
        # 返回result
        return result


class TestSnakeCurve(Scene):
    def construct(self):
        num = 6
        snakecurve_group = [SnakeCurve() for i in range(num)]
        # 实例化了一次之后这个实例在循环里面被调用时候应该使用copy方法复制一份
        # 不然的话循环里面重复调用同一个实例化的物体势必会出问题
        # 非要在循环里面重复调用同一个类的话应该重复实例化多次再加进去

        for i in range(num):
            snakecurve_group[i].order = i + 2
            snakecurve_group[i].init_points()  # 原来父类里面写了怎么连线的方法，，大意了没有闪
            # 下一次一定会好好看父类的！
            snakecurve_group[i].init_colors()
            # snakecurve_group[i].set_stroke(width=4)  # 设置曲线宽度为固定值
            snakecurve_group[i].scale(1.3)  # 进行缩放

        self.play(Create(snakecurve_group[0]))
        self.wait()

        for i in range(len(snakecurve_group) - 1):
            self.play(
                ReplacementTransform(snakecurve_group[i], snakecurve_group[i + 1])
            )
        self.wait()
