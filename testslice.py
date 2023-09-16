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
