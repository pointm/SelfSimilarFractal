from manim import *
from functools import reduce
import random
import itertools as it

class JaggedCurvePiece(VMobject):
    '''
        类 JaggedCurvePiece是 VMobject 的子类。
        
        这其中有一个方法insert_n_curves
        - 如果曲线片段没有任何曲线，就设置它的点为一个零向量。
        - 获取曲线片段的锚点，即每个曲线的起点和终点。
        - 在锚点数组中，根据 n 的值，均匀地选取 n + len(anchors) 个索引。
        - 用选取的锚点作为新的角点，重新设置曲线片段的点。

        这样，就可以在原来的曲线片段中插入 n 个新的曲线，使得曲线片段看起来更加锯齿化。
    '''

    def insert_n_curves(self, n):
        if self.get_num_curves() == 0:
            self.set_points(np.zeros((1, 3)))
        anchors = self.get_anchors()
        indices = np.linspace(
            0, len(anchors) - 1, n + len(anchors)
        ).astype('int')
        self.set_points_as_corners(anchors[indices])


class TestCircular(Scene):

    def construct(self):
        
        s = Square()

        JaggedCurvePiece.insert_n_curves(self, 3)

        
            





        
