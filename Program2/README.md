# Linear Filtering and Edge Detection: 
==============================================================================<br>
Title:&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;
Linear Filtering and Edge Detection<br>
Date:&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;
10/24/2016<br>
Author:&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;Ryu Muthui<br>
Description:&emsp;&nbsp;Implement linear filtering and edge detection, observe changes in smoothing
==============================================================================<br>

## <a href="https://github.com/Coderaulic/Computer_Vision/blob/master/Program1/Program2.cpp">Linear Filtering and Edge Detection</a>:

<strong>Goal</strong>:<br> Implement linear filtering and edge detection and observe the differences in edge detection for different levels of smoothing.
Note that all of the computations in this program will use greyscale images (not color), so it will use the “grey” byte,
rather than “red,” “green,” and “blue.”

<strong>Linear Filtering</strong>:<br>
Linear Filtering (convolution) can be expressed using the following equation:<br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;
![convolve](https://cloud.githubusercontent.com/assets/10789046/24433617/d76ec8de-13de-11e7-9f91-7bb7869d484c.jpg)<br>
Note that (x1, x2) and (y1, y2) describe the extent of the kernel in the x and y dimensions, respectively.

<strong>Iterative Smoothing</strong>:<br>
An effect similar to smoothing with a Gaussian can be achieved by repeatedly smoothing with small kernels such as:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;
![k1](https://cloud.githubusercontent.com/assets/10789046/24433921/fb31b4fa-13e0-11e7-8e68-ce1877bf5ace.jpg)<br>
Implement image smoothing by convolving an image with both of the above kernels repeatedly.

<strong>Edge Detection</strong>:<br>
Compute the image gradients in the x and y directions (separately) by convolving with the following kernels:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;
![k2](https://cloud.githubusercontent.com/assets/10789046/24433922/fb3bfca8-13e0-11e7-8e32-7020bb20e262.jpg)<br>
Note that convolution with the gradient kernels and edge detection are performed only once after all of the
smoothing has been performed. Detect the edges in the image by finding pixels where the gradient magnitude is at
least 10.0 and the magnitude is at a maximum along the direction of the gradient. This is non-maxima suppression.
It will need to perform interpolation on the gradient magnitudes to do the non-maxima suppression correctly.
During the non-maxima suppression, if at a border pixel and one (or more) of the pixels to interpolate from is
outside the image, it should use the closest location inside the image for that pixel.

Program flow:<br>
![flow](https://cloud.githubusercontent.com/assets/10789046/24433981/6a01c906-13e1-11e7-8a7c-b44414a69d61.jpg)<br>

Original Image:<br>
![original](https://cloud.githubusercontent.com/assets/10789046/24434121/625db934-13e2-11e7-9501-193877fad73f.jpg)<br>

Smoothing Applied at x0, x2, x5, x10:<br>
![smooth0](https://cloud.githubusercontent.com/assets/10789046/24434198/f0a4f64e-13e2-11e7-9f54-80e53ba5d3c2.jpg)
![smooth2](https://cloud.githubusercontent.com/assets/10789046/24434196/f0a3781e-13e2-11e7-918f-af9797f14e72.jpg)
![smooth5](https://cloud.githubusercontent.com/assets/10789046/24434197/f0a47692-13e2-11e7-8fe5-4f1d7bdef0b4.jpg)
![smooth10](https://cloud.githubusercontent.com/assets/10789046/24434195/f09f9168-13e2-11e7-9443-d6a06f9a07d8.jpg)

Gradient Magnitude Images of Gmag, Gx, Gy:<br>
![gmag](https://cloud.githubusercontent.com/assets/10789046/24434288/85d99ae4-13e3-11e7-9450-0bec51efba51.jpg)
![gx](https://cloud.githubusercontent.com/assets/10789046/24434292/87a6449e-13e3-11e7-9c3e-962960717114.jpg)
![gy](https://cloud.githubusercontent.com/assets/10789046/24434293/87c3cf6e-13e3-11e7-82ca-074a1b8cbd1e.jpg)

Edge detection at x0, x2, x5, x10:<br>
![edges0](https://cloud.githubusercontent.com/assets/10789046/24434375/f79dffa8-13e3-11e7-8339-29c2c6286f2b.jpg)
![edges2](https://cloud.githubusercontent.com/assets/10789046/24434376/f7a116fc-13e3-11e7-82fa-1873dbdf58ad.jpg)
![edges5](https://cloud.githubusercontent.com/assets/10789046/24434377/f7a375aa-13e3-11e7-8a55-7235c959588d.jpg)
![edges10](https://cloud.githubusercontent.com/assets/10789046/24434374/f79dfb70-13e3-11e7-8b63-05b877bea343.jpg)




