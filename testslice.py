from time import sleep

# from manim import *
from manimlib import *
from functools import reduce
import random
import itertools as it


# class SquareExample(Scene):
#     def construct(self):
#         # 创建一个 VMobject
#         vmob = VMobject()
#         # 将四个点设置为角点，形成一个正方形
#         vmob.set_points_as_corners([UP, RIGHT, DOWN, LEFT, UP])
#         # 将 VMobject 添加到场景中
#         self.play(Create(vmob))

FRAME_WIDTH = 10


def poly(x, coefs):
    return sum(coefs[k] * x**k for k in range(len(coefs)))


def dpoly(x, coefs):
    return sum(k * coefs[k] * x ** (k - 1) for k in range(1, len(coefs)))


def roots_to_coefficients(roots):
    n = len(list(roots))
    return [
        ((-1) ** (n - k)) * sum(np.prod(tup) for tup in it.combinations(roots, n - k))
        for k in range(n)
    ] + [1]


def find_root(func, dfunc, seed=complex(1, 1), tol=1e-8, max_steps=100):
    # Use newton's method
    last_seed = np.inf
    for n in range(max_steps):
        if abs(seed - last_seed) < tol:
            break
        last_seed = seed
        seed = seed - func(seed) / dfunc(seed)
    return seed


def coefficients_to_roots(coefs):
    if len(coefs) == 0:
        return []
    elif coefs[-1] == 0:
        return coefficients_to_roots(coefs[:-1])
    roots = []
    # Find a root, divide out by (x - root), repeat
    for i in range(len(coefs) - 1):
        root = find_root(
            lambda x: poly(x, coefs),
            lambda x: dpoly(x, coefs),
        )
        roots.append(root)
        new_reversed_coefs, rem = np.polydiv(coefs[::-1], [1, -root])
        coefs = new_reversed_coefs[::-1]
    return roots


ROOT_COLORS_DEEP = ["#440154", "#3b528b", "#21908c", "#5dc963", "#29abca"]


class NewtonFractal(Mobject):
    # CONFIG = {
    #     "shader_folder": "newton_fractal",
    #     "shader_dtype": [
    #         ("point", np.float32, (3,)),
    #     ],
    #     "colors": ROOT_COLORS_DEEP,
    #     "coefs": [1.0, -1.0, 1.0, 0.0, 0.0, 1.0],
    #     "scale_factor": 1.0,
    #     "offset": ORIGIN,
    #     "n_steps": 30,
    #     "julia_highlight": 0.0,
    #     "max_degree": 5,
    #     "saturation_factor": 0.0,
    #     "opacity": 1.0,
    #     "black_for_cycles": False,
    #     "is_parameter_space": False,
    # }
    # data = Mobject.set_points(UL, DL)
    def __init__(self, plane, **kwargs):
        self.shader_folder = "newton_fractal"
        self.shader_dtype = [
            ("point", np.float32, (3,)),
        ]
        self.colors = ROOT_COLORS_DEEP
        self.coefs = [1.0, -1.0, 1.0, 0.0, 0.0, 1.0]
        self.scale_factor = 1.0
        self.offset = ORIGIN
        self.n_steps = 30
        self.julia_highlight = 0.0
        self.max_degree = 5
        self.saturation_factor = 0.0
        self.opacity = 1.0
        self.black_for_cycles = False
        self.is_parameter_space = False
        # self.data = self.set_points([UL, DL, UR, DR])

        super().__init__(**kwargs)
        super().__init__(
            scale_factor=plane.get_x_unit_size(),
            offset=plane.n2p(0),
            **kwargs,
        )
        self.replace(plane, stretch=True)

    def init_data(self):
        self.set_points([UL, DL, UR, DR])

    def init_uniforms(self):
        super().init_uniforms()
        self.set_colors(self.colors)
        self.set_julia_highlight(self.julia_highlight)
        self.set_coefs(self.coefs)
        self.set_scale(self.scale_factor)
        self.set_offset(self.offset)
        self.set_n_steps(self.n_steps)
        self.set_saturation_factor(self.saturation_factor)
        self.set_opacity(self.opacity)
        self.uniforms["black_for_cycles"] = float(self.black_for_cycles)
        self.uniforms["is_parameter_space"] = float(self.is_parameter_space)

    def set_colors(self, colors):
        self.uniforms.update(
            {
                f"color{n}": np.array(color_to_rgba(color))
                for n, color in enumerate(colors)
            }
        )
        return self

    def set_julia_highlight(self, value):
        self.uniforms["julia_highlight"] = value

    def set_coefs(self, coefs, reset_roots=True):
        # 将 coefs 中的系数转换为复数
        full_coefs = [*coefs] + [0] * (self.max_degree - len(coefs) + 1)
        coefs = list(map(complex, full_coefs))

        # 将复数系数存储在 uniforms 属性中
        self.uniforms.update(
            {
                f"coef{n}": np.array([coef.real, coef.imag], dtype=np.float64)
                for n, coef in enumerate(coefs)
            }
        )

        # 如果 reset_roots 为 True，则重新计算并设置 mobject 的根
        if reset_roots:
            self.set_roots(coefficients_to_roots(coefs), False)

        # 返回 mobject 本身
        return self

    def set_roots(self, roots, reset_coefs=True):
        # 将 roots 中的根转换为复数
        full_roots = [*roots] + [0] * (self.max_degree - len(roots))
        roots = list(map(complex, full_roots))

        # 将复数根存储在 uniforms 属性中
        self.uniforms["n_roots"] = float(len(roots))
        self.uniforms.update(
            {
                f"root{n}": np.array([root.real, root.imag], dtype=np.float64)
                for n, root in enumerate(roots)
            }
        )

        # 如果 reset_coefs 为 True，则重新计算并设置 mobject 的系数
        if reset_coefs:
            self.set_coefs(roots_to_coefficients(roots), False)

        # 返回 mobject 本身
        return self

    def set_scale(self, scale_factor):
        self.uniforms["scale_factor"] = scale_factor
        return self

    def set_offset(self, offset):
        self.uniforms["offset"] = np.array(offset)
        return self

    def set_n_steps(self, n_steps):
        self.uniforms["n_steps"] = float(n_steps)
        return self

    def set_saturation_factor(self, saturation_factor):
        self.uniforms["saturation_factor"] = float(saturation_factor)
        return self

    def set_opacities(self, *opacities):
        for n, opacity in enumerate(opacities):
            self.uniforms[f"color{n}"][3] = opacity
        return self

    def set_opacity(self, opacity, recurse=True):
        self.set_opacities(*len(self.roots) * [opacity])
        return self


class MentionFatouSetsAndJuliaSets(Scene):
    colors = [RED_E, BLUE_E, TEAL_E, MAROON_E]

    def construct(self):
        # Introduce terms
        f_group, j_group = self.get_fractals()
        f_name, j_name = VGroup(
            Text("Fatou set"),
            Text("Julia set"),
        )
        f_name.next_to(f_group, UP, MED_LARGE_BUFF)
        j_name.next_to(j_group, UP, MED_LARGE_BUFF)

        self.play(Write(j_name), GrowFromCenter(j_group))
        self.wait()
        self.play(Write(f_name), *map(GrowFromCenter, f_group))
        self.wait()

        # Define Fatou set
        fatou_condition = self.get_fatou_condition()
        fatou_condition.set_width(FRAME_WIDTH - 1)
        fatou_condition.center().to_edge(UP, buff=1.0)
        lhs, arrow, rhs = fatou_condition
        f_line = Line(LEFT, RIGHT)
        f_line.match_width(fatou_condition)
        f_line.next_to(fatou_condition, DOWN)
        f_line.set_stroke(WHITE, 1)

        self.play(FadeOut(j_name, RIGHT), FadeOut(j_group, RIGHT), Write(lhs))
        self.wait()
        for words in lhs[-1]:
            self.play(Indicate(words, buff=0, time_width=1.5))
        self.play(Write(arrow))
        self.play(
            LaggedStart(
                FadeTransform(f_name.copy(), rhs[1][:8]), FadeIn(rhs), lag_ratio=0.5
            )
        )
        self.wait()

        # Show Julia set
        otherwise = Text("Otherwise...")
        otherwise.next_to(rhs, DOWN, LARGE_BUFF)
        j_condition = Tex("$z_0 \\in$", " Julia set", " of $f$")
        j_condition.match_height(rhs)
        j_condition.next_to(otherwise, DOWN, LARGE_BUFF)

        j_group.set_height(4.0)
        j_group.to_edge(DOWN)
        j_group.set_x(-1.0)
        j_name = j_condition.get_part_by_tex("Julia set")
        j_underline = Underline(j_name, buff=0.05)
        j_underline.set_color(YELLOW)
        arrow = Arrow(
            j_name.get_bottom(),
            j_group.get_right(),
            path_arc=-45 * DEGREES,
        )
        arrow.set_stroke(YELLOW, 5)

        julia_set = j_group[0]
        julia_set.update()
        julia_set.suspend_updating()
        julia_copy = julia_set.copy()
        julia_copy.clear_updaters()
        julia_copy.set_colors(self.colors)
        julia_copy.set_julia_highlight(0)

        mover = f_group[:-4]
        mover.generate_target()
        mover.target.match_width(rhs)
        mover.target.next_to(rhs, UP, MED_LARGE_BUFF)
        mover.target.shift_onto_screen(buff=SMALL_BUFF)

        self.play(
            Create(f_line),
            FadeOut(f_name),
            MoveToTarget(mover),
        )
        self.play(Write(otherwise), FadeIn(j_condition, 0.5 * DOWN))
        self.wait()
        self.play(
            Create(j_underline),
            Create(arrow),
            FadeIn(j_group[1]),
            FadeIn(julia_copy),
        )
        self.play(
            GrowFromPoint(julia_set, julia_set.get_corner(UL), run_time=2),
            julia_copy.animate.set_opacity(0.2),
        )
        self.wait()

    def get_fractals(self, jy=1.5, fy=-2.5):
        coefs = roots_to_coefficients([-1.5, 1.5, 1j, -1j])
        n = len(coefs) - 1
        colors = self.colors
        f_planes = VGroup(*(self.get_plane() for x in range(n)))
        f_planes.arrange(RIGHT, buff=LARGE_BUFF)
        plusses = [Tex("+") for _ in range(n - 1)]  # Tex("+").replicate(n - 1)
        f_group = Group(*it.chain(*zip(f_planes, plusses)))
        f_group.add(f_planes[-1])
        f_group.arrange(RIGHT)
        fatou = Group(
            *(
                NewtonFractal(f_plane, coefs=coefs, colors=colors)
                for f_plane in f_planes
            )
        )
        for i, fractal in enumerate(fatou):
            opacities = n * [0.2]
            opacities[i] = 1
            fractal.set_opacities(*opacities)
        f_group.add(*fatou)
        f_group.set_y(fy)

        j_plane = self.get_plane()
        j_plane.set_y(jy)
        julia = NewtonFractal(j_plane, coefs=coefs, colors=5 * [GREY_A])
        julia.set_julia_highlight(1e-3)
        j_group = Group(julia, j_plane)

        for fractal, plane in zip((*fatou, julia), (*f_planes, j_plane)):
            fractal.plane = plane
            fractal.add_updater(
                lambda m: m.set_offset(m.plane.get_center())
                .set_scale(m.plane.get_x_unit_size())
                .replace(m.plane)
            )

        fractals = Group(f_group, j_group)
        return fractals

    def get_plane(self):
        plane = ComplexPlane(
            (-2, 2),
            (-2, 2),
            background_line_style={"stroke_width": 1, "stroke_color": GREY},
        )
        plane.set_height(2)
        plane.set_opacity(0)
        box = SurroundingRectangle(plane, buff=0)
        box.set_stroke(WHITE, 1)
        plane.add(box)
        return plane

    def get_fatou_condition(self):
        zn = MathTex(
            "z_0",
            "\\overset{f}{\\longrightarrow}",
            "z_1",
            "\\overset{f}{\\longrightarrow}",
            "z_2",
            "\\overset{f}{\\longrightarrow}",
            "\\dots",
            "\\longrightarrow",
        )
        words = VGroup(
            Tex("Stable fixed point"),
            Tex("Stable cycle"),
            Tex("$\\infty$"),
        )
        words.arrange(DOWN, aligned_edge=LEFT)
        brace = Brace(words, LEFT)
        zn.next_to(brace, LEFT)
        lhs = VGroup(zn, brace, words)

        arrow = MathTex("\\Rightarrow")
        arrow.scale(2)
        arrow.next_to(lhs, RIGHT, MED_LARGE_BUFF)
        rhs = MathTex("z_0 \\in", " \\text{Fatou set of $f$}")
        rhs.next_to(arrow, RIGHT, buff=MED_LARGE_BUFF)

        result = VGroup(lhs, arrow, rhs)

        return result


class TestScene(Scene):
    colors = [RED_E, BLUE_E, TEAL_E, MAROON_E]

    def get_plane(self):
        plane = ComplexPlane(
            # (-2, 2),
            # (-2, 2),
            # background_line_style={"stroke_width": 1, "stroke_color": GREY},
        )
        plane.set_height(2)
        plane.set_opacity(0)
        box = SurroundingRectangle(plane, buff=0)
        box.set_stroke(WHITE, 1)
        plane.add(box)
        return plane

    def get_fractals(self, jy=1.5, fy=-2.5):
        coefs = roots_to_coefficients([-1.5, 1.5, 1j, -1j])
        n = len(coefs) - 1
        colors = self.colors
        f_planes = VGroup(*(self.get_plane() for x in range(n)))
        f_planes.arrange(RIGHT, buff=LARGE_BUFF)
        plusses = [Tex("+") for _ in range(n - 1)]  # Tex("+").replicate(n - 1)
        f_group = Group(*it.chain(*zip(f_planes, plusses)))
        f_group.add(f_planes[-1])
        f_group.arrange(RIGHT)
        fatou = Group(
            *(
                NewtonFractal(f_plane, coefs=coefs, colors=colors)
                for f_plane in f_planes
            )
        )
        for i, fractal in enumerate(fatou):
            opacities = n * [0.2]
            opacities[i] = 1
            fractal.set_opacities(*opacities)
        f_group.add(*fatou)
        f_group.set_y(fy)

        j_plane = self.get_plane()
        j_plane.set_y(jy)
        julia = NewtonFractal(j_plane, coefs=coefs, colors=5 * [GREY_A])
        julia.set_julia_highlight(1e-3)
        j_group = Group(julia, j_plane)

        for fractal, plane in zip((*fatou, julia), (*f_planes, j_plane)):
            fractal.plane = plane
            fractal.add_updater(
                lambda m: m.set_offset(m.plane.get_center())
                .set_scale(m.plane.get_x_unit_size())
                .replace(m.plane)
            )

        fractals = Group(f_group, j_group)
        return fractals

    def construct(self):
        # Introduce terms
        f_group, j_group = self.get_fractals()
        f_name, j_name = VGroup(
            Text("Fatou set"),
            Text("Julia set"),
        )
        f_name.next_to(f_group, UP, MED_LARGE_BUFF)
        j_name.next_to(j_group, UP, MED_LARGE_BUFF)

        self.play(Write(j_name), GrowFromCenter(j_group))
        self.wait()
        self.play(Write(f_name), *map(GrowFromCenter, f_group))
        self.wait()

        # Define Fatou set
        # fatou_condition = self.get_fatou_condition()
        # fatou_condition.set_width(FRAME_WIDTH - 1)
        # fatou_condition.center().to_edge(UP, buff=1.0)
        # lhs, arrow, rhs = fatou_condition
        # f_line = Line(LEFT, RIGHT)
        # f_line.match_width(fatou_condition)
        # f_line.next_to(fatou_condition, DOWN)
        # f_line.set_stroke(WHITE, 1)

        self.play(FadeOut(j_name, RIGHT), FadeOut(j_group, RIGHT))  # , Write(lhs))
        self.wait()
        # for words in lhs[-1]:
        #     self.play(Indicate(words, buff=0, time_width=1.5))
        # self.play(Write(arrow))
        # self.play(
        #     LaggedStart(
        #         FadeTransform(f_name.copy(), rhs[1][:8]), FadeIn(rhs), lag_ratio=0.5
        #     )
        # )
        # self.wait()


if __name__ == "__main__":
    roots = coefficients_to_roots(
        [1.0, -1.0, 1.0, 0.0, 0.0, 1.0]
    )  # 传入系数矩阵，解方程z^5 + z^2 - z + 1 = 0，并且获得根
    print([x**5 + x**2 - x**1 + 1 for x in roots])
