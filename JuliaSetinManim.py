from manim import *
from colour import Color

config.frame_width = 6
config.frame_height = 6


class JuliaSet(Scene):
    def construct(self):
        perc_bond = 3

        # 创建一个复平面对象
        plane = ComplexPlane(
            x_range = (-2, 2, 0.5), # x轴的范围和刻度
            y_range = (-1.5, 1.5, 0.5), # y轴的范围和刻度
            background_line_style = {"stroke_width": 1}, # 背景线条的样式
        )
        # 添加坐标轴和标签
        plane.add_coordinates()
        # 定义一个复数c作为常数
        c = complex(-0.1, 0.65)
        # 定义一个复变函数f(z)作为迭代规则
        f = lambda z: z**2 + c
        # 定义一个颜色映射函数，根据迭代次数来返回颜色
        def get_color(n):
            colors = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]
            return colors[n % len(colors)]
        
        def get_color_out(n):
            colors = [BLUE_A, BLUE_B, BLUE_C, BLUE_D]
            return colors[n % len(colors)]

        # 定义一个函数，看一个数离另一个数够不够近
        # 这个函数用来搜寻茱莉亚集合边界上的点
        # def is_near(x, y, eps):
        #     return math.isclose(x, y, rel_tol=eps)


        # 定义一个函数，根据复数z是否属于茱莉亚集来返回一个点对象
        def get_dot(z, bound, max_iter):
            # 设置一个界限值和最大迭代次数
            # bound = 40
            # max_iter = 20
            # 初始化迭代次数和当前值
            n = 0
            zn = z
            # 进行迭代，直到超过界限或达到最大次数
            while abs(zn) <= bound and n < max_iter:
                zn = f(zn)
                n += 1
            # 如果没有超过界限，说明z属于茱莉亚集，用彩色表示
            if n == max_iter and abs(zn) < bound:
                flag = get_color(n)
            # 否则，用黑色表示
            elif n == max_iter :#and abs(abs(zn)-bound)<bound*perc_bond:
                flag = WHITE
            else:
                flag = get_color_out(n)
            # 根据z在复平面上的位置和颜色创建一个点对象，并返回
            dot = Dot(plane.n2p(z), color = flag, radius = 0.05)
            return dot
        
        # 我想要茱莉亚集边界附近的点
        def get_exdot(z, bound, max_iter):
            # 初始化迭代次数和当前值
            n = 0
            zn = z
            # 进行迭代，直到超过界限或达到最大次数
            while n < max_iter and abs(abs(zn)-bound)<=bound*perc_bond:
                zn = f(zn)
                n += 1
            # 如果刚好在界限附近，那么就输出为白色
            if abs(abs(zn)-bound)<=bound*perc_bond:
                flag = WHITE
            # 否则，用黑色
            else:
                flag = get_color_out(n)
            # 根据z在复平面上的位置和颜色创建一个点对象，并返回
            dot = Dot(plane.n2p(z), color = flag, radius = 0.05)
            return dot

        # 创建一个包含所有点对象的VGroup对象
        dots = VGroup()
        # 遍历复平面上的每个点，根据是否属于茱莉亚集添加到VGroup对象中
        for x in np.arange(-2, 2, 0.1):
            for y in np.arange(-2, 2, 0.1):
                z = complex(x, y)
                dot = get_dot(z,4,20)#最大的边界值为4，迭代次数为20
                dots.add(dot)
        
        

        # 将复平面和所有点对象添加到场景中，并播放动画
        self.add(plane,dots)
        # self.add(plane)
        # self.play(Create(dots), run_time = 2)
        # self.wait()

def is_julia(z, c, ita, lim):
    flag = 0
    # while flag < ita and abs(z) < lim:
    #     z = z**2 + c
    #     flag += 1
    # if flag == ita:
    #     return True
    # else:
    #     return False
    for i in range(ita):
        z = z**2 +c
        flag = i
        if abs(z) > lim:
            return False, flag, abs(z)#原来这里自带一个Break啊， 惊了
    return True, flag, abs(z)
    


class NewJulia(Scene):      
    def construct(self):
        ita_num = 20#迭代次数为20
        limitation = 4#迭代的最大幅值为4
        c = complex(-0.12,0.65)#茱莉亚集合里面的常数为0.12 + 0.15j
        dots = VGroup()#预创建要显示的列表
        plane = ComplexPlane()

        # z = complex(6.13,0.15)

        # 创建一个包含所有点对象的VGroup对象
        dots = VGroup()
        # 遍历复平面上的每个点，根据是否属于茱莉亚集添加到VGroup对象中
        step = 0.02
        for x in np.arange(-2, 2, step):
            for y in np.arange(-2, 2, step):
                z = complex(x, y)
                [julia, num, amplitude] = is_julia(z,c,ita_num,limitation)
                if julia:#最大的边界值为4，迭代次数为20
                    dots.add(Square(
                                    side_length = 4*step/8,
                                    #plane.n2p(z), 
                                    color = Color(hue = amplitude/limitation, saturation = 1, luminance = 0.5),                                 
                                #  color = Color(hue = 1, saturation = amplitude/limitation, luminance = 0.5),                                 
                                #  color = Color(hue = 0.3, saturation = 1, luminance = amplitude/limitation),                                 
                                    fill_opacity=0.8).move_to([x,y,0]))
                else:
                    dots.add(Square(
                                    side_length = 4*step/8,
                                    #plane.n2p(z),
                                    color = Color(hue = num/ita_num, saturation = 1, luminance = 0.5),
                                    # color = Color(hue = 1, saturation = num/ita_num, luminance = 0.5),
                                    # color = Color(hue = 0.4, saturation = 1, luminance = num/ita_num),
                                    # radius = 0.02,
                                    fill_opacity=0.5).move_to([x,y,0]))
        # test = Tex(str(is_julia(z, c, ita_num, limitation)))
        # test = str(test)
        # print(is_julia)
        self.add(plane)
        self.add(dots)
        # self.play(AnimationGroup(*[FadeIn(s, run_time = 0.2) for s in dots], lag_ratio = 0.001)) 本来想用这个的，但是结果没想到渲染太慢了，而且渲染结果并不好看
        self.play(Create(dots,run_time = 2))
        self.wait()
        # self.wait()需要注意的是，在step = 0.01之后再减小步长并没有很大的作用
        # 因为这个东西的分辨率的话，也与迭代的函数的迭代次数与边界boundary相关
        return super().construct()
