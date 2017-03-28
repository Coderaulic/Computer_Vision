//------------------------------------------------------------------------------
//                      Program3.cpp
//------------------------------------------------------------------------------
// CSS487:		Computer Vision – Program 3
// Programmer:		Ryu Muthui
// Creation Date:	11/01/2016
// Date last modified:	11/09/2016
// Purpose:		This program explores graphic manipulation using the
//			OpenCV library. The program has two main parts.
//			Part 1: Computes a color histogram and creates an
//			overlay image combining the two images.
//			The most common color in the foreground is
//			then replaced with the background image.
//			Part 2: Manipulates an image using OpenCV methods.
//------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

//------------------------------------------------------------------------------
// int main() -		The main entry into the program.
// Preconditions:	foreground.jpg and background.jpg exists and are correctly
//			formatted JPG images.
// Postconditions:	Creates two images of "overlay.jpg" and "output.jpg" as 
//			output images. "overlay.jpg" is created using the foreground
//			and background image. "output.jpg" is created using the
//			background image, with OpenCV methods applied.
// Notes:		Following the formatting of OpenCV with color,
//			[Blue, Green, Red] = [0, 1, 2] in terms of position.
//			In addition, the program waits for a key press between each
//			image display, press the space bar to continue.
//------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
	//--------------------------------------------------------------------------
	// Read two images from the disk.
	Mat foreground = imread("foreground.jpg");
	Mat background = imread("background.jpg");

	//--------------------------------------------------------------------------
	// Part 1:
	// Create a histogram
	const int size = 4;
	int bucketSize = 256 / size;
	int dims[] = { size, size, size };
	Mat hist(3, dims, CV_32S, Scalar::all(0));

	//--------------------------------------------------------------------------
	// Scan through foreground image and update the histogram bucket with the
	// most common color.
	for (int rows = 0; rows < foreground.rows; rows++) {
		for (int cols = 0; cols < foreground.cols; cols++) {
			int blue = foreground.at<Vec3b>(rows, cols)[0] / bucketSize;
			int green = foreground.at<Vec3b>(rows, cols)[1] / bucketSize;
			int red = foreground.at<Vec3b>(rows, cols)[2] / bucketSize;
			hist.at<int>(blue, green, red) += 1;
		}
	}

	//--------------------------------------------------------------------------
	// Scan through the histogram bin with the most votes
	int max = 0;
	Vec3i histPosition = { 0, 0, 0 };
	for (int blue = 0; blue < size; blue++) {
		for (int green = 0; green < size; green++) {
			for (int red = 0; red < size; red++) {
				if (hist.at<int>(blue, green, red) > max) {
					max = hist.at<int>(blue, green, red);
					histPosition[0] = blue;
					histPosition[1] = green;
					histPosition[2] = red;
				}
			}
		}
	}

	//--------------------------------------------------------------------------
	// Convert to color in the middle of the bucket
	histPosition[0] = histPosition[0] * bucketSize + bucketSize / 2;
	histPosition[1] = histPosition[1] * bucketSize + bucketSize / 2;
	histPosition[2] = histPosition[2] * bucketSize + bucketSize / 2;

	//--------------------------------------------------------------------------
	// Set up the min/max threshhold for each positions
	int threshMinBlue = histPosition[0] - bucketSize;
	int threshMaxBlue = histPosition[0] + bucketSize;
	int threshMinGreen = histPosition[1] - bucketSize;
	int threshMaxGreen = histPosition[1] + bucketSize;
	int threshMinRed = histPosition[2] - bucketSize;
	int threshMaxRed = histPosition[2] + bucketSize;

	//--------------------------------------------------------------------------
	// Scan through the forground image. When the most common color pixel is
	// found, replace it with the background image pixel.
	for (int rows = 0; rows < foreground.rows; rows++) {
		for (int cols = 0; cols < foreground.cols; cols++) {

			// Get the current pixel color value
			int  bVal = foreground.at<Vec3b>(rows, cols)[0];
			int  gVal = foreground.at<Vec3b>(rows, cols)[1];
			int  rVal = foreground.at<Vec3b>(rows, cols)[2];

			// If within the threshhold values, replace the pixels
			if ((bVal >= threshMinBlue && bVal <= threshMaxBlue) &&
				(gVal >= threshMinGreen && gVal <= threshMaxGreen) &&
				(rVal >= threshMinRed && rVal <= threshMaxRed)) {
				int modRows = rows % background.rows;
				int modCols = cols % background.cols;
				foreground.at<Vec3b>(rows, cols)[0] = 
					background.at<Vec3b>(modRows, modCols)[0];
				foreground.at<Vec3b>(rows, cols)[1] = 
					background.at<Vec3b>(modRows, modCols)[1];
				foreground.at<Vec3b>(rows, cols)[2] = 
					background.at<Vec3b>(modRows, modCols)[2];
			}
		}
	}

	//--------------------------------------------------------------------------
	// Save "overlay.jpg" image and display it to screen.
	imwrite("overlay.jpg", foreground);
	namedWindow("Overlay Image");
	imshow("Overlay Image", foreground);
	waitKey(0);

	//--------------------------------------------------------------------------
	// Part 2: Manipulates "background.jpg" using OpenCV methods.
	//	   Flips horizontally, change to grayScale image, smooths with
	//	   GaussianBlur, and gets an edge image with Canny.
	flip(background, background, 1);
	cvtColor(background, background, COLOR_BGR2GRAY);
	GaussianBlur(background, background, Size(7, 7), 2.0, 2.0, BORDER_DEFAULT);
	Canny(background, background, 20, 60, 3, false);
	imwrite("output.jpg", background);
	namedWindow("Output Image");
	imshow("Output Image", background);
	waitKey(0);
	return 0;
}