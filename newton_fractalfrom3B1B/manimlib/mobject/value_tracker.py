from __future__ import annotations

import numpy as np

from manimlib.mobject.mobject import Mobject
from manimlib.utils.iterables import listify


class ValueTracker(Mobject):
    """
    记录一个数值（不在画面中显示）

    传入的 ``value`` 为初始数值
    """
    CONFIG = {
        "value_type": np.float64,
    }

    def __init__(self, value: float | complex = 0, **kwargs):
        self.value = value
        super().__init__(**kwargs)

    def init_data(self) -> None:
        super().init_data()
        self.data["value"] = np.array(
            listify(self.value),
            ndmin=2,
            dtype=self.value_type,
        )

    def get_value(self) -> float | complex:
        '''获取当前值'''
        result = self.data["value"][0, :]
        if len(result) == 1:
            return result[0]
        return result

    def set_value(self, value: float | complex):
        '''将值设为 ``value``'''
        self.data["value"][0, :] = value
        return self

    def increment_value(self, d_value: float | complex) -> None:
        '''将值增加 ``d_value``'''
        self.set_value(self.get_value() + d_value)


class ExponentialValueTracker(ValueTracker):
    """
    以指数形式变化的存值器

    传入的 ``value`` 为初始数值
    """

    def get_value(self) -> float | complex:
        '''获取当前存的值'''
        return np.exp(ValueTracker.get_value(self))

    def set_value(self, value: float | complex):
        '''将值设为 ``value``'''
        return ValueTracker.set_value(self, np.log(value))


class ComplexValueTracker(ValueTracker):
    '''记录一个复数数值

    传入的 ``value`` 为初始数值
    '''
    CONFIG = {
        "value_type": np.complex128
    }
