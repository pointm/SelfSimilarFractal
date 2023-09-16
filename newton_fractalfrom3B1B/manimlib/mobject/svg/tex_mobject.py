from __future__ import annotations

from functools import reduce
import operator as op
import re

from manimlib.constants import BLACK, WHITE
from manimlib.constants import DOWN, LEFT, RIGHT, UP
from manimlib.constants import FRAME_WIDTH
from manimlib.constants import MED_LARGE_BUFF, MED_SMALL_BUFF, SMALL_BUFF
from manimlib.mobject.geometry import Line
from manimlib.mobject.svg.svg_mobject import SVGMobject
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.utils.config_ops import digest_config
from manimlib.utils.tex_file_writing import display_during_execution
from manimlib.utils.tex_file_writing import get_tex_config
from manimlib.utils.tex_file_writing import tex_to_svg_file

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from colour import Color
    from typing import Iterable, Union

    ManimColor = Union[str, Color]


SCALE_FACTOR_PER_FONT_POINT = 0.001


class SingleStringTex(SVGMobject):
    """单个字符串渲染出的 SVGMobject"""
    CONFIG = {
        "height": None,
        "fill_opacity": 1.0,
        "stroke_width": 0,
        "svg_default": {
            "color": WHITE,
        },
        "path_string_config": {
            "should_subdivide_sharp_curves": True,
            "should_remove_null_curves": True,
        },
        "font_size": 48,
        "alignment": "\\centering",
        "math_mode": True,
        "organize_left_to_right": False,
    }

    def __init__(self, tex_string: str, **kwargs):
        """只传入一个字符串``tex_string``"""
        assert isinstance(tex_string, str)
        self.tex_string = tex_string
        super().__init__(**kwargs)

        if self.height is None:
            self.scale(SCALE_FACTOR_PER_FONT_POINT * self.font_size)
        if self.organize_left_to_right:
            self.organize_submobjects_left_to_right()

    @property
    def hash_seed(self) -> tuple:
        return (
            self.__class__.__name__,
            self.svg_default,
            self.path_string_config,
            self.tex_string,
            self.alignment,
            self.math_mode
        )

    def get_file_path(self) -> str:
        full_tex = self.get_tex_file_body(self.tex_string)
        with display_during_execution(f"Writing \"{self.tex_string}\""):
            file_path = tex_to_svg_file(full_tex)
        return file_path

    def get_tex_file_body(self, tex_string: str) -> str:
        new_tex = self.get_modified_expression(tex_string)
        if self.math_mode:
            new_tex = "\\begin{align*}\n" + new_tex + "\n\\end{align*}"

        new_tex = self.alignment + "\n" + new_tex

        tex_config = get_tex_config()
        return tex_config["tex_body"].replace(
            tex_config["text_to_replace"],
            new_tex
        )

    def get_modified_expression(self, tex_string: str) -> str:
        """将对齐参数与传入的字符串拼接，并且处理特殊的字符串"""
        return self.modify_special_strings(tex_string.strip())

    def modify_special_strings(self, tex: str) -> str:
        """处理特殊不合法的字符串"""
        tex = tex.strip()
        should_add_filler = reduce(op.or_, [
            # Fraction line needs something to be over
            tex == "\\over",
            tex == "\\overline",
            # Makesure sqrt has overbar
            tex == "\\sqrt",
            tex == "\\sqrt{",
            # Need to add blank subscript or superscript
            tex.endswith("_"),
            tex.endswith("^"),
            tex.endswith("dot"),
        ])
        if should_add_filler:
            filler = "{\\quad}"
            tex += filler

        should_add_double_filler = reduce(op.or_, [
            tex == "\\overset",
            # TODO: these can't be used since they change
            # the latex draw order.
            # tex == "\\frac", # you can use \\over as a alternative 
            # tex == "\\dfrac",
            # tex == "\\binom",
        ])
        if should_add_double_filler:
            filler = "{\\quad}{\\quad}"
            tex += filler

        if tex == "\\substack":
            tex = "\\quad"

        if tex == "":
            tex = "\\quad"

        # To keep files from starting with a line break
        if tex.startswith("\\\\"):
            tex = tex.replace("\\\\", "\\quad\\\\")

        tex = self.balance_braces(tex)

        # Handle imbalanced \left and \right
        num_lefts, num_rights = [
            len([
                s for s in tex.split(substr)[1:]
                if s and s[0] in "(){}[]|.\\"
            ])
            for substr in ("\\left", "\\right")
        ]
        if num_lefts != num_rights:
            tex = tex.replace("\\left", "\\big")
            tex = tex.replace("\\right", "\\big")

        for context in ["array"]:
            begin_in = ("\\begin{%s}" % context) in tex
            end_in = ("\\end{%s}" % context) in tex
            if begin_in ^ end_in:
                # Just turn this into a blank string,
                # which means caller should leave a
                # stray \\begin{...} with other symbols
                tex = ""
        return tex

    def balance_braces(self, tex: str) -> str:
        """匹配大括号"""
        num_unclosed_brackets = 0
        for i in range(len(tex)):
            if i > 0 and tex[i - 1] == "\\":
                # So as to not count '\{' type expressions
                continue
            char = tex[i]
            if char == "{":
                num_unclosed_brackets += 1
            elif char == "}":
                if num_unclosed_brackets == 0:
                    tex = "{" + tex
                else:
                    num_unclosed_brackets -= 1
        tex += num_unclosed_brackets * "}"
        return tex

    def get_tex(self) -> str:
        return self.tex_string

    def organize_submobjects_left_to_right(self):
        self.sort(lambda p: p[0])
        return self


class Tex(SingleStringTex):
    """用于生成 LaTeX 公式（align 环境）"""
    CONFIG = {
        "arg_separator": "",
        "isolate": [],
        "tex_to_color_map": {},
    }

    def __init__(self, *tex_strings: str, **kwargs):
        """可传入多个 ``tex_strings``
        
        - ``arg_separator`` 表示每两个字符串之间的字符，默认为空格
        - ``isolate`` 列表中放有想要单独拆开的字符串，可以不用手动拆开
        - ``tex_to_color_map`` 为一个字典，会根据其中的键自动拆开字符串用于上色
        """
        digest_config(self, kwargs)
        self.tex_strings = self.break_up_tex_strings(tex_strings)
        full_string = self.arg_separator.join(self.tex_strings)
        super().__init__(full_string, **kwargs)
        self.break_up_by_substrings()
        self.set_color_by_tex_to_color_map(self.tex_to_color_map)

        if self.organize_left_to_right:
            self.organize_submobjects_left_to_right()

    def break_up_tex_strings(self, tex_strings: Iterable[str]) -> Iterable[str]:
        """根据传入的 tex_to_color_map 再次拆开 tex_strings"""
        # Separate out any strings specified in the isolate
        # or tex_to_color_map lists.
        substrings_to_isolate = [*self.isolate, *self.tex_to_color_map.keys()]
        if len(substrings_to_isolate) == 0:
            return tex_strings
        patterns = (
            "({})".format(re.escape(ss))
            for ss in substrings_to_isolate
        )
        pattern = "|".join(patterns)
        pieces = []
        for s in tex_strings:
            if pattern:
                pieces.extend(re.split(pattern, s))
            else:
                pieces.append(s)
        return list(filter(lambda s: s, pieces))

    def break_up_by_substrings(self):
        """重新组织子物体，``tex_string`` 中每个子字符串为一个子物体"""
        if len(self.tex_strings) == 1:
            submob = self.copy()
            self.set_submobjects([submob])
            return self
        new_submobjects = []
        curr_index = 0
        config = dict(self.CONFIG)
        config["alignment"] = ""
        for tex_string in self.tex_strings:
            tex_string = tex_string.strip()
            if len(tex_string) == 0:
                continue
            sub_tex_mob = SingleStringTex(tex_string, **config)
            num_submobs = len(sub_tex_mob)
            if num_submobs == 0:
                continue
            new_index = curr_index + num_submobs
            sub_tex_mob.set_submobjects(self[curr_index:new_index])
            new_submobjects.append(sub_tex_mob)
            curr_index = new_index
        self.set_submobjects(new_submobjects)
        return self

    def get_parts_by_tex(
        self,
        tex: str,
        substring: bool = True,
        case_sensitive: bool = True
    ) -> VGroup:
        def test(tex1, tex2):
            if not case_sensitive:
                tex1 = tex1.lower()
                tex2 = tex2.lower()
            if substring:
                return tex1 in tex2
            else:
                return tex1 == tex2

        return VGroup(*filter(
            lambda m: isinstance(m, SingleStringTex) and test(tex, m.get_tex()),
            self.submobjects
        ))

    def get_part_by_tex(self, tex: str, **kwargs) -> SingleStringTex | None:
        all_parts = self.get_parts_by_tex(tex, **kwargs)
        return all_parts[0] if all_parts else None

    def set_color_by_tex(self, tex: str, color: ManimColor, **kwargs):
        """给 ``tex`` 上颜色为 ``color``，注意此时 ``tex`` 要独立存在，否则会给包含 ``tex`` 的也上色"""
        self.get_parts_by_tex(tex, **kwargs).set_color(color)
        return self

    def set_color_by_tex_to_color_map(
        self,
        tex_to_color_map: dict[str, ManimColor],
        **kwargs
    ):
        """根据 ``texs_to_color_map`` 上色，同样，会给包含键的全部上色，不会自动拆分"""
        for tex, color in list(tex_to_color_map.items()):
            self.set_color_by_tex(tex, color, **kwargs)
        return self

    def index_of_part(self, part: SingleStringTex, start: int = 0) -> int:
        return self.submobjects.index(part, start)

    def index_of_part_by_tex(self, tex: str, start: int = 0, **kwargs) -> int:
        """根据 tex 获取在子物体中的下标"""
        part = self.get_part_by_tex(tex, **kwargs)
        return self.index_of_part(part, start)

    def slice_by_tex(
        self,
        start_tex: str | None = None,
        stop_tex: str | None = None,
        **kwargs
    ) -> VGroup:
        if start_tex is None:
            start_index = 0
        else:
            start_index = self.index_of_part_by_tex(start_tex, **kwargs)

        if stop_tex is None:
            return self[start_index:]
        else:
            stop_index = self.index_of_part_by_tex(stop_tex, start=start_index, **kwargs)
            return self[start_index:stop_index]

    def sort_alphabetically(self) -> None:
        """根据字典序给子物体排序"""
        self.submobjects.sort(key=lambda m: m.get_tex())

    def set_bstroke(self, color: ManimColor = BLACK, width: float = 4):
        self.set_stroke(color, width, background=True)
        return self


class TexText(Tex):
    """ 用于生成 LaTeX 文字，默认每行之间居中
    
    传入的两个字符串之间无分隔 (即 ``arg_separator=""``)
    """
    CONFIG = {
        "math_mode": False,
        "arg_separator": "",
    }


class BulletedList(TexText):
    """项目列表"""
    CONFIG = {
        "buff": MED_LARGE_BUFF,
        "dot_scale_factor": 2,
        "alignment": "",
    }

    def __init__(self, *items: str, **kwargs):
        """ 支持多个字符串，每个一行；也支持一个字符串，使用 LaTeX 的换行（\\\\）"""
        line_separated_items = [s + "\\\\" for s in items]
        TexText.__init__(self, *line_separated_items, **kwargs)
        for part in self:
            dot = Tex("\\cdot").scale(self.dot_scale_factor)
            dot.next_to(part[0], LEFT, SMALL_BUFF)
            part.add_to_back(dot)
        self.arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=self.buff
        )

    def fade_all_but(self, index_or_string: int | str, opacity: float = 0.5) -> None:
        """把除了 ``index_or_string`` 之外的不透明度均设为 ``opacity``
        
        ``index_or_string`` 可以传入子物体的下标，也可以传入一个字符串
        """
        arg = index_or_string
        if isinstance(arg, str):
            part = self.get_part_by_tex(arg)
        elif isinstance(arg, int):
            part = self.submobjects[arg]
        else:
            raise Exception("Expected int or string, got {0}".format(arg))
        for other_part in self.submobjects:
            if other_part is part:
                other_part.set_fill(opacity=1)
            else:
                other_part.set_fill(opacity=opacity)


class TexFromPresetString(Tex):

    CONFIG = {
        # To be filled by subclasses
        "tex": None,
        "color": None,
    }

    def __init__(self, **kwargs):
        digest_config(self, kwargs)
        Tex.__init__(self, self.tex, **kwargs)
        self.set_color(self.color)


class Title(TexText):
    """标题"""
    CONFIG = {
        "scale_factor": 1,
        "include_underline": True,
        "underline_width": FRAME_WIDTH - 2,
        # This will override underline_width
        "match_underline_width_to_text": False,
        "underline_buff": MED_SMALL_BUFF,
    }

    def __init__(self, *text_parts: str, **kwargs):
        """``include_underline=True`` 会添加下划线（默认添加）
        ``underline_width`` 下划线的长度（默认屏幕宽 - 2 个单位）
        ``match_underline_width_to_text=True`` 时将下划线的长度和文字匹配（默认为 False）
        """
        TexText.__init__(self, *text_parts, **kwargs)
        self.scale(self.scale_factor)
        self.to_edge(UP)
        if self.include_underline:
            underline = Line(LEFT, RIGHT)
            underline.next_to(self, DOWN, buff=self.underline_buff)
            if self.match_underline_width_to_text:
                underline.match_width(self)
            else:
                underline.set_width(self.underline_width)
            self.add(underline)
            self.underline = underline
