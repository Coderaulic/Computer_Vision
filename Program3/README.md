# Color Histogram: 
==============================================================================<br>
Title:&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;
Color Histogram<br>
Date:&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;
11/09/2016<br>
Author:&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;Ryu Muthui<br>
Description:&emsp;Compute a color histogram and overlay matching colors of an image's background
==============================================================================<br>

## <a href="https://github.com/Coderaulic/Computer_Vision/blob/master/Program3/Program3.cpp">Color Histogram</a>:

<strong>Goal</strong>:<br> Modify an image similar to “green screen” technique.
Pixels of a selected color will be replaced with pixels from a second image. 
For further information on the general idea, see: <a href="http://en.wikipedia.org/wiki/Chroma_key">Chroma Key</a>

<strong>Compute a color histogram</strong>:<br>
1) Read two images from the disk.
2) Use a <a href="https://en.wikipedia.org/wiki/Color_histogram">color histogram</a> to find the most common color in the foreground image.<br>
The histogram should be a three-dimensional matrix of integers. 
   - int dims[] = {size, size, size};&emsp;&emsp;&nbsp;&emsp;&emsp;&nbsp;&emsp;&emsp;&nbsp;&emsp;&emsp;&nbsp;// size is a constant - the # of buckets in each dimension<br>
Mat hist(3, dims, CV_32S, Scalar::all(0)); &emsp;&emsp;&nbsp;&emsp;&nbsp;&nbsp;// 3D histogram of integers initialized to zero<br>
   - To create the histogram, loop through the foreground image and assign each pixel to a histogram bucket, incremented by one.<br>To decide which bucket to increment, you divide the color value by (256 / size):
      - int bucketSize = 256 / size;
      - int r = red / bucketSize;
      - int g = green / bucketSize;
      - int b = blue / bucketSize;
3) Find the histogram bin with the most “votes” by looping over all three dimensions.<br>
   If the bin with the most votes is [r, g, b], then the most common color is approximately:<br>
   - int cRed = r * bucketSize + bucketSize/2;
   - int cGreen = g * bucketSize + bucketSize/2;
   - int cBlue = b * bucketSize + bucketSize/2;

<strong>Create the overlay output</strong>:<br>
Replace every pixel in the foreground image that is close to the most common color (no more than bucketSize
away in all three color bands) with the corresponding pixel from the background image (same row and
column, unless the background image is too small). If the background image is too small, start over from the
start of the background image. Display the resulting image on the screen.

Foreground Image:<br>
![foreground](https://cloud.githubusercontent.com/assets/10789046/24435244/fb87c6b2-13e8-11e7-874a-352386416e80.jpg)
Background Image:<br>
![background](https://cloud.githubusercontent.com/assets/10789046/24435246/fb8ee6c2-13e8-11e7-9389-633ea6bb05c2.jpg)
Combined Results:<br>
![overlay](https://cloud.githubusercontent.com/assets/10789046/24435245/fb8de592-13e8-11e7-82e4-c68bc825304f.jpg)

