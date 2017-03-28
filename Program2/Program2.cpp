//------------------------------------------------------------------------------
//                      Program2.cpp
//------------------------------------------------------------------------------
// CSS487:		Computer Vision – Program 2
// Programmer:		Ryu Muthui
// Creation Date:	10/10/2016
// Date last modified:	10/24/2016
// Purpose:		This program outputs a smoothed image and an edge
//			image of the original input image. An image "test.gif"
//			assumed to be in the same directory as the program
//			executes from, taking in 1 parameter value which
//			specifies how many times to apply the smoothing.
//			It is also assumed that Image.h and Image.lib are
//			inlucded when running the program.
// Limitations:		Currently the program only supports x and y kernel
//			images of 1x3 and 3x1 that are hard coded.
//------------------------------------------------------------------------------
#include "Image.h"
#include <iostream>
#include <math.h>
using namespace std;

// Point object to represent a point in an (x, y) coordinate system
struct Point {
	float x = 0.0f;
	float y = 0.0f;
};

//------------------------------------------------------------------------------
// Function that returns a float value used when determining an edge pixel for
// when calculating the edge image.
float bilinearInterpolation(Point point, const Image &gradientIm);

//------------------------------------------------------------------------------
// int main() - 	The main entry into the program.
// Preconditions: 	test.gif exists and is a correctly formatted GIF image
// Postconditions: 	Creates two images of "smooth.gif" and "edge.gif" as output
//			images, applying the smoothing n times.
//------------------------------------------------------------------------------
int main(int argc, char *argv[]) {
	//--------------------------------------------------------------------------
	// Check if inputs are valid before program starts
	if (argc != 2) {
		cerr << "Incorrect number of parameters supplied." << endl;
		cerr << "Usage: " << argv[0] << ", Smoothing value." << endl;
		return -1;
	}

	if (atoi(argv[1]) < 0) {
		cerr << "Smooth count should be 0 or greater." << endl;
		return -1;
	}

	//--------------------------------------------------------------------------
	// Variables and Images
	int smoothnessFactor = atoi(argv[1]);
	Image input("test.gif");
	int rowMax = input.getRows();
	int colMax = input.getCols();
	Image smoothed(rowMax, colMax);
	Image tempSmooth(rowMax, colMax);
	Image gradientX(rowMax, colMax);
	Image gradientY(rowMax, colMax);
	Image gradientMag(rowMax, colMax);
	Image edges(rowMax, colMax);

	// Kernel images
	Image x_Kernel(1, 3);
	Image y_Kernel(3, 1);
	Image x_EdgeKernel(1, 3);
	Image y_EdgeKernel(3, 1);

	// [1/4, 1/2/, 1/4] left, mid, right
	x_Kernel.setFloat(0, 0, 0.25f);
	x_Kernel.setFloat(0, 1, 0.5f);
	x_Kernel.setFloat(0, 2, 0.25f);

	// [1/4] top
	// [1/2] mid
	// [1/4] bottom
	y_Kernel.setFloat(0, 0, 0.25f);
	y_Kernel.setFloat(1, 0, 0.5f);
	y_Kernel.setFloat(2, 0, 0.25f);

	// [-1, 0, 1] left, mid, right
	x_EdgeKernel.setFloat(0, 0, -1.0f);
	x_EdgeKernel.setFloat(0, 1, 0.0f);
	x_EdgeKernel.setFloat(0, 2, 1.0f);

	// [-1] top
	// [ 0] mid
	// [ 1] bottom
	y_EdgeKernel.setFloat(0, 0, -1.0f);
	y_EdgeKernel.setFloat(1, 0, 0.0f);
	y_EdgeKernel.setFloat(2, 0, 1.0f);

	//--------------------------------------------------------------------------
	// Convert the input image to float image
	for (int row = 0; row < rowMax; row++) {
		for (int col = 0; col < colMax; col++) {
			smoothed.setFloat(row, col, input.getPixel(row, col).grey);
		}
	}

	//--------------------------------------------------------------------------
	// Convolve: Apply smoothing on Sx
	for (int i = 0; i < smoothnessFactor; i++) {
		for (int row = 0; row < rowMax; row++) {
			for (int col = 0; col < colMax; col++) {

				float left = 0.0f, x_mid = 0.0f, right = 0.0f, x_Smooth = 0.0f;
				x_mid = smoothed.getFloat(row, col);

				// Check left bounds and set values accordingly
				if (col - 1 < 0) {
					left = x_mid;
				}
				else {
					left = smoothed.getFloat(row, col - 1);
				}

				// Check right bounds and set values accordingly
				if (col + 1 >= colMax) {
					right = x_mid;
				}
				else {
					right = smoothed.getFloat(row, col + 1);
				}

				x_Smooth = (left * x_Kernel.getFloat(0, 2)) +
					(x_mid * x_Kernel.getFloat(0, 1)) +
					(right * x_Kernel.getFloat(0, 0));
				tempSmooth.setFloat(row, col, x_Smooth);
			}
		}
		smoothed = tempSmooth;
	}

	//--------------------------------------------------------------------------
	// Convolve: Apply smoothing on Sy
	for (int i = 0; i < smoothnessFactor; i++) {
		for (int row = 0; row < rowMax; row++) {
			for (int col = 0; col < colMax; col++) {

				float top = 0.0, y_mid = 0.0, bottom = 0.0, y_Smooth = 0.0;
				y_mid = smoothed.getFloat(row, col);

				// Check upper bounds and set values accordingly
				if (row - 1 < 0)
					top = y_mid;
				else
					top = smoothed.getFloat(row - 1, col);

				// Check lower bounds and set values accordingly
				if (row + 1 >= rowMax)
					bottom = y_mid;
				else
					bottom = smoothed.getFloat(row + 1, col);

				y_Smooth = (top * y_Kernel.getFloat(2, 0)) +
					(y_mid * y_Kernel.getFloat(1, 0)) +
					(bottom * y_Kernel.getFloat(0, 0));
				tempSmooth.setFloat(row, col, y_Smooth);
			}
		}
		smoothed = tempSmooth;
	}

	//--------------------------------------------------------------------------
	// Create Gradient in x, using edge kernel image for x
	for (int row = 0; row < rowMax; row++) {
		for (int col = 0; col < colMax; col++) {

			float left = 0.0, x_mid = 0.0, right = 0.0, x_Smooth = 0.0;
			x_mid = smoothed.getFloat(row, col);

			// Check left bounds and set values accordingly
			if (col - 1 < 0)
				left = x_mid;
			else
				left = smoothed.getFloat(row, col - 1);

			// Check right bounds and set values accordingly
			if (col + 1 >= colMax)
				right = x_mid;
			else
				right = smoothed.getFloat(row, col + 1);

			x_Smooth = (left * x_EdgeKernel.getFloat(0, 2)) +
				(x_mid * x_EdgeKernel.getFloat(0, 1)) +
				(right * x_EdgeKernel.getFloat(0, 0));
			tempSmooth.setFloat(row, col, x_Smooth);
		}
	}
	gradientX = tempSmooth;

	//--------------------------------------------------------------------------
	// Create Gradient in y, using edge kernel image for y
	for (int row = 0; row < rowMax; row++) {
		for (int col = 0; col < colMax; col++) {

			float top = 0.0, y_mid = 0.0, bottom = 0.0, y_Smooth = 0.0;
			y_mid = smoothed.getFloat(row, col);

			// Check upper bounds and set values accordingly
			if (row - 1 < 0)
				top = y_mid;
			else
				top = smoothed.getFloat(row - 1, col);

			// Check lower bounds and set values accordingly
			if (row + 1 >= rowMax)
				bottom = y_mid;
			else
				bottom = smoothed.getFloat(row + 1, col);

			y_Smooth = (top * y_EdgeKernel.getFloat(2, 0)) +
				(y_mid * y_EdgeKernel.getFloat(1, 0)) +
				(bottom * y_EdgeKernel.getFloat(0, 0));
			tempSmooth.setFloat(row, col, y_Smooth);
		}
	}
	gradientY = tempSmooth;

	//--------------------------------------------------------------------------
	// Create the Gradient Magnitude Image
	for (int row = 0; row < rowMax; row++) {
		for (int col = 0; col < colMax; col++) {
			float gX = gradientX.getFloat(row, col);
			float gY = gradientY.getFloat(row, col);
			float magnitude = sqrtf(gX * gX + gY * gY);
			gradientMag.setFloat(row, col, magnitude);
		}
	}

	//--------------------------------------------------------------------------
	// Create the smoothed image
	for (int row = 0; row < rowMax; row++) {
		for (int col = 0; col < colMax; col++) {
			smoothed.setGrey(row, col, static_cast<byte>(smoothed.getFloat(row, col)));
		}
	}
	smoothed.writeGreyImage("smooth.gif");

	//--------------------------------------------------------------------------
	// Create the edge image
	// Calls: bilinearInterpolation()
	for (int row = 0; row < rowMax; row++) {
		for (int col = 0; col < colMax; col++) {

			Point r, p;
			float gMag = gradientMag.getFloat(row, col);

			if (gMag >= 10.0f) {
				r.x = col + ((gradientX.getFloat(row, col)) / gradientMag.getFloat(row, col));
				r.y = row + ((gradientY.getFloat(row, col)) / gradientMag.getFloat(row, col));
				p.x = col - ((gradientX.getFloat(row, col)) / gradientMag.getFloat(row, col));
				p.y = row - ((gradientY.getFloat(row, col)) / gradientMag.getFloat(row, col));

				float gR = bilinearInterpolation(r, gradientMag);
				float gP = bilinearInterpolation(p, gradientMag);

				// Set edge pixel to white when conditions met
				if ((gMag > gR) && (gMag > gP)) {
					edges.setFloat(row, col, 255);
				}
			}
		}
	}
	edges.writeFloatImage("edges.gif");
	return 0;
}

//------------------------------------------------------------------------------
// float bilinearInterpolation() - Returns a float value
// Preconditions: 	takes in a gradient magnitude image to check from, and a
//			a point object with the cordinates to check a against.
// Postconditions: 	Returns a float value to be used for comparison when
//			determining the edge
//------------------------------------------------------------------------------
float bilinearInterpolation(Point point, const Image &gradientIm) {

	// Variables for interpolation calculation
	float xInterpolate = floor(point.x);
	float yInterpolate = floor(point.y);
	float alpha = (point.y - yInterpolate);
	float beta = (point.x - xInterpolate);
	float yPlus1 = yInterpolate + 1.0f;
	float xPlus1 = xInterpolate + 1.0f;
	float val = 0.0f;

	// Check for left bounds and set values accordingly
	if (xInterpolate < 0.0f)
		xInterpolate = 0;

	if (xInterpolate >= gradientIm.getCols())
		xInterpolate = gradientIm.getCols() - 1.0f;

	if (xPlus1 >= gradientIm.getCols())
		xPlus1 = gradientIm.getCols() - 1.0f;

	if (xPlus1 <= -1.0f)
		xPlus1 = 0.0f;

	// Check for left bounds and set values accordingly
	if (yInterpolate < 0.0f)
		yInterpolate = 0.0f;

	if (yInterpolate >= gradientIm.getRows())
		yInterpolate = gradientIm.getRows() - 1.0f;

	if (yPlus1 >= gradientIm.getRows())
		yPlus1 = gradientIm.getRows() - 1.0f;

	if (yPlus1 <= -1.0f)
		yPlus1 = 0.0f;

	val = ((1.0f - alpha) * (1.0f - beta) * gradientIm.getFloat((int)yInterpolate, (int)xInterpolate) +
		alpha * (1.0f - beta) * gradientIm.getFloat((int)yPlus1, (int)xInterpolate) +
		(1.0f - alpha) * beta * gradientIm.getFloat((int)yInterpolate, (int)xPlus1) +
		alpha * beta * gradientIm.getFloat((int)yPlus1, (int)xPlus1));
	return val;
}