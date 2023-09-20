from manim import *
import random

config.frame_width = 8
# config.frame_height = 8


def is_julia(z, c, ita, lim):
    flag = 0
    for i in range(ita):
        z = z**2 + c
        flag = i
        if abs(z) > lim:
            return False, flag, abs(z)  # 原来这里自带一个Break啊， 惊了
    return True, flag, abs(z)


class TestSceneForImageMobject(Scene):
    def calccomplex(self, x_range, y_range, xunitnumber, yunitnumber, xorder, yorder):
        x_length = abs(x_range[1] - x_range[0])
        y_length = abs(y_range[1] - y_range[0])
        x_unit = x_length / xunitnumber
        y_unit = y_length / yunitnumber

        return complex(
            xorder * x_unit + x_range[0], -yorder * y_unit + y_range[1]
        ) + complex(x_unit / 2, -y_unit / 2)

    def construct(self):
        ita_num = 20  # 迭代次数为20
        limitation = 4  # 迭代的最大幅值为4
        c = complex(-0.12, 0.65)  # 茱莉亚集合里面的常数为0.12 + 0.15j
        x_range = [-3, 3]
        y_range = [-3, 3]
        unitnumber = 1000

        xunit = yunit = unitnumber
        totalunit = xunit * yunit

        plane = ComplexPlane()
        self.add(plane)

        randomcolor = [random.randint(0, 255) for _ in range(xunit * yunit * 3)]
        randomcolor = np.array(randomcolor).reshape(xunit, yunit, 3)

        coorcomplex = []
        # dots = VGroup()
        for i in range(xunit):
            for j in range(yunit):
                complexvar = self.calccomplex(x_range, y_range, xunit, yunit, i, j)
                [julia, num, amplitude] = is_julia(complexvar, c, ita_num, limitation)
                if julia:
                    randomcolor[i][j] = [0,0,0]
                coorcomplex.append(complexvar)
                # dots.add(Dot(plane.n2p(complexvar), radius=0.1))

        coorcomplex = np.array(coorcomplex).reshape(xunit, yunit)

        # self.add(dots)

        image = ImageMobject(np.uint8(randomcolor))
        image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        image.height = abs(x_range[0]-x_range[1])
        image.move_to([np.mean(x_range),np.mean(y_range),0])
        self.add(image.set_opacity(1))
        return super().construct()


class GetPoints(Scene):
    def construct(self):
        plane = NumberPlane()
        points = plane.get_ordered_pairs()
        print(points)
        self.add(plane)
