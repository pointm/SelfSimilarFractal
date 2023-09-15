from manim import *
from functools import reduce
import random
import itertools as it

class SelfSimilarFractal(VMobject):
    '''
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

    '''
    order = 5#一个表示分形的阶数的整数，用于控制分形的递归深度。阶数越高，分形越复杂。默认值是5。
    num_subparts = 3#一个表示每个分形部分包含多少个子部分的整数，用于控制分形的分支数量。默认值是3。
    height = 4#一个表示分形图形的高度的浮点数，用于控制分形的大小。默认值是4。
    colors = [RED, WHITE]#一个表示分形图形的颜色渐变的列表，用于控制分形的外观。默认值是[RED, WHITE]
    stroke_width = 1  #一个表示分形图形的描边宽度的浮点数，用于控制分形的边缘粗细。默认值是1。
    # 注意，这不是正确的方式来设置描边宽度，应该使用set_stroke()方法。Not the right way to assign stroke width
    fill_opacity = 1  #一个表示分形图形的填充不透明度的浮点数，用于控制分形的透明度。默认值是1。
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
            subparts = [
                lower_order.copy()
                for x in range(self.num_subparts)
            ]
            self.arrange_subparts(*subparts)
            result = VGroup(*subparts)

        result.set_height(self.height)
        result.center()
        return result

    def get_seed_shape(self):
        raise Exception("Not implemented")

    def arrange_subparts(self, *subparts):
        raise Exception("Not implemented")


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

    def construct(self):
        sierpinskistage = VGroup()
        sierpinskistage.add(Sierpinski().get_seed_shape())#得到种子图像
        sierpinskistage.set_opacity(0.5).set_color([RED, YELLOW, BLUE])#顺便调整种子图像的透明度与颜色
        
        # 调用sierpinsk_ita开始迭代
        # 不写一整个循环的主要原因是要手动调整三角形的大小，，，
        # 如果一直只缩放0.75倍的话，后面迭代的话就会超出屏幕
        for i in range(3):
            self.play(Transform(sierpinskistage, self.sierpinsk_ita(sierpinskistage).scale(0.75)))
        for i in range(3):
            self.play(Transform(sierpinskistage, self.sierpinsk_ita(sierpinskistage).scale(0.55)))


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
     # 定义一个函数，接受一个vgroup对象作为参数
    def pent_ita(self, vgroup, num):#函数默认应该加上一个self函数
        # 将原先的vgroup复制四份，装入列表中
        var = [vgroup.copy() for i in range(num)]
        # 把四个迭代的vgroup的位置放到位
        PentagonalFractal().arrange_subparts(*var)#这时候就把四个四边形的位置放到位了
        #不需要设置任何的原函数参数返回
        # 把原先的列表转化为VGroup，不然的话不好调用后面的move_to和scale等方法
        vgroup = VGroup()
        vgroup.add(*var)
        # 把VGroup移动到原点
        vgroup.move_to(ORIGIN)
        # 返回VGroup对象
        return vgroup
    
    def construct(self):
        penstage = VGroup()
        var = PentagonalFractal.get_seed_shape(self)
        penstage.add(var)

        
        for i in range(3):
            self.play(Transform(penstage, 
                                TestPentagonal.pent_ita(self, penstage, 5)
                                .scale(0.575)))



        
