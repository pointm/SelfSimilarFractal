# JuliaSetinManim
This is a repository that visualize Julia Set by using `manim` in python
The python version is 3.11.2, and the `manimCE` version is 0.17.3

# Usage 
Firstly, make sure that you have installed python, `manim` and `ffmpeg` correctly.You can see the installation in https://docs.manim.community/en/stable/

Then use the following command:

```
manim .\JuliaSetinManim.py -p NewJulia --disable_caching
```
![image] (NewJulia_ManimCE_v0.17.3.png)

It might takes some time to render this Scene, you can decrease the time by setting the quality of output video lower.