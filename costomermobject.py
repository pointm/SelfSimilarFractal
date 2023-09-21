from manim import *

# import random
import matplotlib.pyplot as plt

# [详细的配色方案如下
#     "Accent",
#     "Blues",
#     "BrBG",
#     "BuGn",
#     "BuPu",
#     "CMRmap",
#     "Dark2",
#     "GnBu",
#     "Greens",
#     "Greys",
#     "OrRd",
#     "Oranges",
#     "PRGn",
#     "Paired",
#     "Pastel1",
#     "Pastel2",
#     "PiYG",
#     "PuBu",
#     "PuBuGn",
#     "PuOr",
#     "PuRd",
#     "Purples",
#     "RdBu",
#     "RdGy",
#     "RdPu",
#     "RdYlBu",
#     "RdYlGn",
#     "Reds",
#     "Set1",
#     "Set2",
#     "Set3",
#     "Spectral",
#     "Wistia",
#     "YlGn",
#     "YlGnBu",
#     "YlOrBr",
#     "YlOrRd",
#     "afmhot",
#     "autumn",
#     "binary",
#     "bone",
#     "brg",
#     "bwr",
#     "cividis",
#     "cool",
#     "coolwarm",
#     "copper",
#     "cubehelix",
#     "flag",
#     "gist_earth",
#     "gist_gray",
#     "gist_heat",
#     "gist_ncar",
#     "gist_stern",
#     "gist_yarg",
#     "gnuplot",
#     "gnuplot2",
#     "gray",
#     "hot",
#     "hsv",
#     "inferno",
#     "jet",
#     "magma",
#     "nipy_spectral",
#     "ocean",
#     "pink",
#     "plasma",
#     "prism",
#     "rainbow",
#     "seismic",
#     "spring",
#     "summer",
#     "tab10",
#     "tab20",
#     "tab20b",
#     "tab20c",
#     "terrain",
#     "twilight",
#     "twilight_shifted",
#     "viridis",
#     "winter",
# ]

config.frame_width = 6
# config.frame_height = 3


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
        offset = [0, 0]
        center = [0 + offset[0], 0 + offset[1]]
        radius = 1.5
        x_range = [center[0] - radius, center[0] + radius]
        y_range = [center[1] - radius, center[1] + radius]
        unitnumber = 1000
        # colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        xunit = yunit = unitnumber
        totalunit = xunit * yunit
        # 创建一个 jet colormap
        cmap = plt.cm.summer
        # 获取 colormap 中的所有颜色
        colors = cmap(np.linspace(0, 1, ita_num)) * 255
        colors = [subcolor[:-1] for subcolor in colors]
        # colors.reverse()

        # 获取每个颜色的 RGB 值
        for i in range(len(colors)):
            print(colors[i])

        plane = ComplexPlane()
        self.add(plane)

        randomcolor = [0 for _ in range(xunit * yunit * 3)]
        randomcolor = np.array(randomcolor).reshape(xunit, yunit, 3)

        coorcomplex = []
        # dots = VGroup()
        for i in range(xunit):
            for j in range(yunit):
                complexvar = self.calccomplex(x_range, y_range, xunit, yunit, i, j)
                [julia, num, amplitude] = is_julia(complexvar, c, ita_num, limitation)
                if julia:
                    randomcolor[i][j] = [0, 0, 0]
                else:
                    randomcolor[i][j] = colors[num % len(colors)]
                coorcomplex.append(complexvar)
                # dots.add(Dot(plane.n2p(complexvar), radius=0.02))

        coorcomplex = np.array(coorcomplex).reshape(xunit, yunit)

        # self.add(dots)

        image = ImageMobject(np.uint8(randomcolor))
        image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        image.height = abs(x_range[0] - x_range[1])
        # image.height = abs(4)
        # image.move_to([np.mean(x_range), np.mean(y_range), 0])
        self.add(image.set_opacity(1))
        return super().construct()


class GetPoints(Scene):
    def construct(self):
        plane = NumberPlane()
        points = plane.get_ordered_pairs()
        print(points)
        self.add(plane)
