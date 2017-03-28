# Linear Transformation: 
==============================================================================<br>
Title:&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;
Linear Transformation<br>
Date:&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;
10/07/2016<br>
Author:&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;Ryu Muthui<br>
Description:&emsp;&nbsp;Applying linear algebra concepts to transform an image
==============================================================================<br>

### <a href="https://github.com/Coderaulic/Computer_Vision/blob/master/Program1/Program1.cpp">Linear Transform</a>:

<strong>Goal</strong>: Use linear algebra to transform an image according to parameters inputs.

A general linear transformation of a two-dimensional image has six parameters. Let’s work in
the x-y plane (rather than row-column). Each point (px, py) could be transformed according to the
parameters (a, b, c, d, e, f) as follows:<br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;
![equation1](https://cloud.githubusercontent.com/assets/10789046/24430204/54c052b8-13ca-11e7-8369-aea06067ae5d.jpg)<br>

In this program, we will use more intuitive variables that have the same range of generality.
Our parameters will be two scale factors (sx and sy), a translation in x and y (tx and ty), a rotation
angle in degrees (θ), and a shear (k). Given these values, we can write a linear transformation as
follows:<br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;
![equation2](https://cloud.githubusercontent.com/assets/10789046/24430463/963a4734-13cb-11e7-962d-80b10cd4a43b.jpg)<br>
Or, more simply:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;
![equation4](https://cloud.githubusercontent.com/assets/10789046/24430465/9642f55a-13cb-11e7-9428-c39f05fbd639.jpg)<br>
where:<br>
![equation3](https://cloud.githubusercontent.com/assets/10789046/24430464/9640e6c0-13cb-11e7-8cc2-1b15edafa3bf.jpg)

You are to write a program that takes these six parameters (sx, sy, tx, ty, θ, and k) as input from the
batch file (in that order) and transforms an image according to the parameters. The output image
should have the same dimensions as the input image. Any point in the output image that doesn’t
have a corresponding point in the input image should be black.


