from manim import *
from colour import Color

config.frame_width = 6
config.frame_height = 6


def is_julia(z, c, ita, lim):
    flag = 0
    for i in range(ita):
        z = z**2 + c
        flag = i
        if abs(z) > lim:
            return False, flag, abs(z)  # 原来这里自带一个Break啊， 惊了
    return True, flag, abs(z)


class NewJulia(Scene):
    def construct(self):
        ita_num = 20  # 迭代次数为20
        limitation = 4  # 迭代的最大幅值为4
        c = complex(-0.12, 0.65)  # 茱莉亚集合里面的常数为0.12 + 0.15j
        dots = VGroup()  # 预创建要显示的列表
        plane = ComplexPlane()

        # z = complex(6.13,0.15)

        # 创建一个包含所有点对象的VGroup对象
        dots = VGroup()
        # 遍历复平面上的每个点，根据是否属于茱莉亚集添加到VGroup对象中
        step = 0.02
        for x in np.arange(-2, 2, step):
            for y in np.arange(-2, 2, step):
                z = complex(x, y)
                [julia, num, amplitude] = is_julia(z, c, ita_num, limitation)
                if julia:  # 最大的边界值为4，迭代次数为20
                    dots.add(
                        Square(
                            side_length=step * 1.1,
                            # color=Color(
                            #     hue=amplitude / limitation, saturation=1, luminance=0.5
                            # ),
                            # color=Color(
                            #     hue=1, saturation=amplitude / limitation, luminance=0.5
                            # ),
                            color=Color(
                                hue=0.3, saturation=1, luminance=amplitude / limitation
                            ),
                            fill_opacity=1,
                        )
                        .move_to([x, y, 0])
                        .set_stroke(width=0)
                    )
                # else:
                #     dots.add(Square(
                #                     side_length = 4*step/8,
                #                     color = Color(hue = num/ita_num, saturation = 1, luminance = 0.5),
                #                     # color = Color(hue = 1, saturation = num/ita_num, luminance = 0.5),
                #                     # color = Color(hue = 0.4, saturation = 1, luminance = num/ita_num),
                #                     # radius = 0.02,
                #                     fill_opacity=0.5).move_to([x,y,0]))
        self.add(plane)
        self.add(dots)
        # self.play(AnimationGroup(*[FadeIn(s, run_time = 0.2) for s in dots], lag_ratio = 0.001)) 本来想用这个的，但是结果没想到渲染太慢了，而且渲染结果并不好看
        self.play(Create(dots, run_time=2, rate_func=lambda t: t))
        self.wait()
        # 在step = 0.01之后再减小步长并没有很大的作用
        # 因为这个东西的分辨率的话，也与迭代的函数的迭代次数与边界boundary相关
        return super().construct()
