//------------------------------------------------------------------------------
//                      Program1.cpp
//------------------------------------------------------------------------------
// CSS487:		Computer Vision – Program 1
// Programmer:		Ryu Muthui
// Creation Date:	10/03/2016
// Date last modified:	10/07/2016
// Purpose:		A program that reads in an image "test.gif" assumed
//			to be in the same dir as the program executes from,
//			taking in 6 parameter values and modifies it via
//			translation, scaling, shearing, and rotating the
//			original image. In addition, bilinear interpolation
//			is applied and returned as an image "output.gif".
//------------------------------------------------------------------------------
#include "Image.h"
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
using namespace std;

//------------------------------------------------------------------------------
// int main() - 	The main entry into the program.
// Preconditions: 	test.gif exists and is a correctly formatted GIF image
// Postconditions: 	Create an image as "output.gif" applying changes w/ the 
//			passed in params of "test.gif"
//------------------------------------------------------------------------------
int main(int argc, char *argv[]) {

	// check inputs are valid before program starts
	if (argc != 7) {
		cerr << "Incorrect number of parameters supplied." << endl;
		cerr << "Usage: (sx, sy, tx, ty, 0, and k)" << endl;
		return -1;
	}

	if (atoi(argv[1]) == 0 || atoi(argv[2]) == 0) {
		cerr << "scale factor can't be 0" << endl;
		return -1;
	}

	struct Point {
		double x = 0;
		double y = 0;
	};

	Point center;

	// Create an image object with the given file
	Image input("test.gif");

	// Create an output image file that will reflect the changes
	Image output(input.getRows(), input.getCols());

	double x_scale, y_scale, x_transform, y_transform, theta, shear;

	sscanf_s(argv[1], "%lf", &x_scale);
	sscanf_s(argv[2], "%lf", &y_scale);
	sscanf_s(argv[3], "%lf", &x_transform);
	sscanf_s(argv[4], "%lf", &y_transform);
	sscanf_s(argv[5], "%lf", &theta);
	sscanf_s(argv[6], "%lf", &shear);

	// calculate the center positions
	center.x = input.getCols() / 2;
	center.y = input.getRows() / 2;

	double xPrime = 0.0;
	double yPrime = 0.0;

	for (int row = 0; row < input.getRows(); row++) {
		for (int col = 0; col < input.getCols(); col++) {

			// calculate (q - t - c) value
			double x1 = col - x_transform - center.x;
			double y1 = row - y_transform - center.y;

			// S^-1
			// | 1/Sx   0  |
			// |  0   1/Sy |
			// calculate S^1 matrix (scale) applied with previous values
			double x2 = (1 / x_scale) * x1;
			double y2 = (1 / y_scale) * y1;

			// K^-1
			// | 1 -k |
			// | 0  1 |
			// calculate K^-1 matrix (shear) applied with previous values
			double x3 = x2 + (-shear * y2);
			double y3 = y2;

			// R^-1
			// |  cos0  sin0 |
			// | -sin0  cos0 |
			// calculate R^-1 matrix (rotation) applied with previous values
			double x4 = (x3 * cos(theta * M_PI / 180)) + (y3 * sin(theta * M_PI / 180));
			double y4 = (x3 * -sin(theta * M_PI / 180)) + (y3 * cos(theta * M_PI / 180));

			// add back the ( + c) center value
			xPrime = x4 + center.x;
			yPrime = y4 + center.y;

			// variables for calculating bilinear interpolation
			int xInterpolate = floor(xPrime); // r
			int yInterpolate = floor(yPrime); // c
			double alpha = (yPrime - yInterpolate);
			double beta = (xPrime - xInterpolate);
			byte red = 0, green = 0, blue = 0;

			// if in valid range of the output.gif image size
			if ((yPrime >= 0 && yPrime < input.getRows()) && (xPrime >= 0 && xPrime < input.getCols())) {

				red = (1 - alpha) * (1 - beta) * input.getPixel(yInterpolate, xInterpolate).red +
					alpha * (1 - beta) * input.getPixel(yInterpolate + 1.0, xInterpolate).red +
					(1 - alpha) * beta * input.getPixel(yInterpolate, xInterpolate + 1.0).red +
					alpha * beta * input.getPixel(yInterpolate + 1.0, xInterpolate + 1.0).red;

				green = (1 - alpha) * (1 - beta) * input.getPixel(yInterpolate, xInterpolate).green +
					alpha * (1 - beta) * input.getPixel(yInterpolate + 1.0, xInterpolate).green +
					(1 - alpha) * beta * input.getPixel(yInterpolate, xInterpolate + 1.0).green +
					alpha * beta * input.getPixel(yInterpolate + 1.0, xInterpolate + 1.0).green;

				blue = (1 - alpha) * (1 - beta) * input.getPixel(yInterpolate, xInterpolate).blue +
					alpha * (1 - beta) * input.getPixel(yInterpolate + 1.0, xInterpolate).blue +
					(1 - alpha) * beta * input.getPixel(yInterpolate, xInterpolate + 1.0).blue +
					alpha * beta * input.getPixel(yInterpolate + 1.0, xInterpolate + 1.0).blue;

				pixel temp;
				temp.red = static_cast<byte>(red);
				temp.green = static_cast<byte>(green);
				temp.blue = static_cast<byte>(blue);
				output.setPixel(row, col, temp);
			}
		}
	}
	output.writeImage("output.gif");
	return 0;
}