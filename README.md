# 使用`manim`渲染茱莉亚集
This repository visualizes Julia Set using Manim, a Python library for creating animated scenes.
这个仓库打算使用python里面的manim库（一种创建关于数学的动画的库）针对分形中的茱莉亚集进行可视化。

I am still working on this project. I want to visualize the Julia Set by VMobject in manim eventually.
工程仍在进行，我最后想用`manim`里面的`VMobject`类针对茱莉亚集进行渲染，现在还只是单纯的在平面上渲染茱莉亚集的像素点，然后拼接在一起而已。

Also, 3B1B's `fractal.py` is based on manimlib or manimgl, and I want to modify them to be more compatible with the manimCE library. And I want to add some test code to `fractal.py`, which is named `fractalsfrom3B1B.py` in this project now,  to make it more readable.
与此同时，3B1B之前所编写的`fractal.py`是基于manimlib或者manimgl编写的，我想要将其进行重构以适配现在的manimCE社区版本，并且在其中加点注释以及测试相关的代码块以方便阅读。更改中的文件在仓库中被命名为`fractalsfrom3B1B.py`。
在完全重构并且实现我想要的功能之后，这个模块可能会被单独的拿出来新作一个仓库罢，当然那肯定都是很久很久之后的事情了。。

![Image text](RenderCover2_ManimCE_v0.17.3.gif)