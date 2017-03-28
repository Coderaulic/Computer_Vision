//------------------------ NeuralNetworkTrainer -----------------------
// Filename:		neural_network_trainer.cpp
// Project Team:	RRVision
// Group Members:	Robert Griswold and Ryu Muthui
// Date:		6 Dec 2016
// Description:		A neural network trainer that can read in images 
//			that are labeled (from start of filename to first
//			peroid. Different keypoint detectors can be used,
//			confidence levels are provided for most likely 
//			and next most likely, and training data can be 
//			loaded from file.
//					
//			Deformable Part Models are optionally used to 
//			optimize the neural network training and testing.
//					
//			Libraries used are boost 1.62.0, open-cv master
//			05 Dec 2016, and opencv_contrib master 05 Dec 2016.
//					
//			Implementation is based largely on Abner Matheus'
//			Dog or Cat Neural Network Application in Open CV
//			Jan 31st, 2016:
//http://picoledelimao.github.io/blog/2016/01/31/is-it-a-cat-or-dog-a-neural-network-application-in-opencv/
//					
//			DPM implementation is based largely on Jiaolong Xu's
//			OpenCV-contrib here:
//https://github.com/opencv/opencv_contrib/tree/master/modules/dpm
//
//			More information here:
//http://docs.opencv.org/3.1.0/d1/d73/tutorial_introduction_to_svm.html
//------------------------------ Includes ----------------------------

#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <fstream>
#include <direct.h>

#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dpm.hpp>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

namespace fs = boost::filesystem;
namespace dpm = cv::dpm;

//------------------------- Configuration ----------------------------
// Feature detector selection
// #define USE_KAZE
#define USE_SURF
// #define USE_SIFT

// time to wait for user to press a key - set to 0 to wait indefinitely
#define WAITKEY_DELAY 30

// minimum size of image for DPM
#define MIN_SIZE 150

// Note: No images will be written to file without this
#define DISPLAY_CONFIDENCE

// Use DPM when reading images for training
#define USE_DPM_TO_TRAIN

// Use DPM when reading images for testing (not recommended)
// #define USE_DPM_TO_TEST

// dpm xml directory. Assumes classnames will match xml name.
std::string dpmModelDirectory = "xml"; 

//-------------------------- Structures ------------------------------
typedef std::vector<std::string>::const_iterator vec_iter;

struct ImageData {
	std::string classname;
	cv::Mat bowFeatures;
};

struct ProbabilityStruct {
	std::string classname;
	float probability;
	std::string nextClassname;
	float nextProbability;
};

//----------------------- Helper Functions ---------------------------

/**
* Get all files in directory (not recursive)
* @param directory Directory where the files are contained
* @return A list containing the file name of all files inside given directory
**/
std::vector<std::string> getFilesInDirectory(const std::string& directory) {
	std::vector<std::string> files;
	fs::path root(directory);
	fs::directory_iterator it_end;
	for (fs::directory_iterator it(root); it != it_end; ++it) {
		if (fs::is_regular_file(it->path())) {
			files.push_back(it->path().string());
		}
	}
	return files;
}

/**
* Extract the class name from a file name
*/
inline std::string getClassName(const std::string& filename) {
	int start = filename.find_last_of('\\') + 1;
	int end = filename.find_first_of('.') - filename.find_last_of('\\') - 1;
	return filename.substr(start, end);
}

/**
* Extract local features for an image
*/
cv::Mat getDescriptors(const cv::Mat& img) {
#ifdef USE_KAZE
	cv::Ptr<cv::KAZE> featureDetector = cv::KAZE::create();
#else
#ifdef USE_SURF
	cv::Ptr<cv::xfeatures2d::SURF> featureDetector = cv::xfeatures2d::SURF::create();
#else
#ifdef USE_SIFT
	cv::Ptr<cv::xfeatures2d::SIFT> featureDetector = cv::xfeatures2d::SIFT::create();
#endif // USE_SIFT
#endif // USE_SURF
#endif // USE_KAZE
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	featureDetector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

	return descriptors;
}

/**
* Creates a rectangle that is the provided rectanlge bounded to the given 
* image dimensions.
*/
cv::Rect boundRectToImg(cv::Rect &theRect, const cv::Mat &theImg) {
	cv::Rect retVal;
	if (theImg.empty()) {
		std::cerr << "Warning: boundRectToImg failed." << std::endl;
		return retVal;
	}

	cv::Point topLeft(theRect.x, theRect.y); // inclusive
	cv::Point bottomRight(theRect.x + theRect.width, theRect.y + theRect.height); // exclusive

	topLeft.x = std::max(topLeft.x, 0);
	topLeft.y = std::max(topLeft.y, 0);
	bottomRight.x = std::min(bottomRight.x, theImg.cols);
	bottomRight.y = std::min(bottomRight.y, theImg.rows);

	retVal = cv::Rect(topLeft, bottomRight);

	return retVal;
}

/**
* Read images from a list of file names and returns, for each read image,
* its class name, its local descriptors, the image mat (greyscale), and the filename.
*/
void readImages(vec_iter begin, vec_iter end, std::function<void(const std::string&,
	const cv::Mat&, const cv::Mat& image, const std::string& filename)> callback) {
	for (auto it = begin; it != end; ++it) {
		std::string filename = *it;
		std::cout << "Reading image " << filename << "..." << std::endl;
		cv::Mat img = cv::imread(filename, 0); // greyscale
		if (img.empty()) {
			std::cerr << "WARNING: Could not read image." << std::endl;
			continue;
		}
		std::string classname = getClassName(filename);
		cv::Mat descriptors = getDescriptors(img);
		callback(classname, descriptors, img, filename);
	}
}

/**
* Read images from a list of file names and returns, for each read image,
* its class name, its local descriptors, a cropped image mat (greyscale) using DPM,
* and the filename.
* 
* This method will create a DPM object detector, use the largest detection
* for the img callback, and optionally display the detected rectangles to the user.
* NOTE: This will only retain the largest DPM object detected currently.
*/
void readImagesWithDPM(vec_iter begin, vec_iter end, std::function<void(const std::string&, 
	const cv::Mat&, const cv::Mat& image, const std::string& filename)> callback) {
	for (auto it = begin; it != end; ++it) {
		std::string filename = *it;
		std::cout << "Reading image " << filename << "..." << std::endl;
		cv::Mat img = cv::imread(filename, 1); // original color
		if (img.empty()) {
			std::cerr << "WARNING: Could not read image." << std::endl;
			continue;
		}
		std::string classname = getClassName(filename);

		if (img.cols >= MIN_SIZE && img.rows >= MIN_SIZE) { // enforce minimum size for dpm
			// load xml
			std::ifstream test(dpmModelDirectory + "\\" + classname + ".xml");
			if (!test.good()) {
				std::cerr << dpmModelDirectory << "\\" << classname << ".xml not found." << std::endl;
				exit(-1);
			}
			test.close();
			cv::Ptr<dpm::DPMDetector> detector = dpm::DPMDetector::create(std::vector<std::string>(1, dpmModelDirectory + "\\" + classname + ".xml"));

			// perform dpm detection
			cv::Mat dpmImg = img.clone();
			std::vector<dpm::DPMDetector::ObjectDetection> ds;
			detector->detect(dpmImg, ds);

			// draw rectangles on a frame and find the largest one
			cv::Scalar color(255, 0, 255); // Magenta //TODO change color based on xml
			cv::Mat frame = img.clone();
			cv::Rect largestRect;
			if (ds.size() > 0) largestRect = ds[0].rect;
			for (unsigned int i = 1; i < ds.size(); i++) {
				// find the largest rectangle area
				if (ds[i].rect.area() > largestRect.area()) {
					largestRect = ds[i].rect;
				}
				else {
					// draw a rectangle that is darker but not used for neural network
					rectangle(frame, ds[i].rect, color - cv::Scalar(150, 150, 150), 2);
				}
			}

			// draw the largest rectangle
			rectangle(frame, largestRect, color, 2);

			// display the frame to the user
			cv::namedWindow("DPM Cascade Detection", 1);
			imshow("DPM Cascade Detection", frame);

			// bound the rectangle to the image and pass img as the largest rect to neural network
			if (ds.size() > 0) {
				largestRect = boundRectToImg(largestRect, img);
				img = img(largestRect);
			}
		}
		else {
			std::cerr << "Waring: Image is too small for DPM assessment." << std::endl;
		}

		cv::cvtColor(img, img, CV_BGR2GRAY, 1);

		cv::Mat descriptors = getDescriptors(img);
		callback(classname, descriptors, img, filename);
	}
}

/**
* Transform a class name into an id
*/
int getClassId(const std::set<std::string>& classes, const std::string& classname) {
	int index = 0;
	
	for (auto it = classes.begin(); it != classes.end(); ++it) {
		if (*it == classname) {
			break;
		}
		++index;
	}
	return index;
}

/**
* Get a binary code associated to a class
*/
cv::Mat getClassCode(const std::set<std::string>& classes, const std::string& classname) {
	cv::Mat code = cv::Mat::zeros(cv::Size((int)classes.size(), 1), CV_32F);
	int index = getClassId(classes, classname);
	code.at<float>(index) = 1;
	return code;
}

/**
* Turn local features into a single bag of words histogram of
* of visual words (a.k.a., bag of words features)
*/
cv::Mat getBOWFeatures(cv::FlannBasedMatcher& flann, const cv::Mat& descriptors,
	int vocabularySize) {
	cv::Mat outputArray = cv::Mat::zeros(cv::Size(vocabularySize, 1), CV_32F);
	std::vector<cv::DMatch> matches;
	flann.match(descriptors, matches);
	for (size_t j = 0; j < matches.size(); j++) {
		int visualWord = matches[j].trainIdx;
		outputArray.at<float>(visualWord)++;
	}
	return outputArray;
}

/**
* Get a trained neural network according to some inputs and outputs
*/
cv::Ptr<cv::ml::ANN_MLP> getTrainedNeuralNetwork(const cv::Mat& trainSamples,
	const cv::Mat& trainResponses) {
	int networkInputSize = trainSamples.cols;
	int networkOutputSize = trainResponses.cols;
	cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();
	std::vector<int> layerSizes = { networkInputSize, networkInputSize / 2,
		networkOutputSize };
	mlp->setLayerSizes(layerSizes);
	mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
	mlp->train(trainSamples, cv::ml::ROW_SAMPLE, trainResponses);
	return mlp;
}

/**
* Receives a column matrix contained the probabilities associated to
* each class and returns the id of column which contains the highest
* probability
*/
int getPredictedClass(const cv::Mat& predictions) {
	float maxPrediction = predictions.at<float>(0);
	int maxPredictionIndex = 0;
	const float* ptrPredictions = predictions.ptr<float>(0);
	for (int i = 0; i < predictions.cols; i++) {
		float prediction = *ptrPredictions++;
		if (prediction > maxPrediction) {
			maxPrediction = prediction;
			maxPredictionIndex = i;
		}
	}
	return maxPredictionIndex;
}

/**
* Print a confusion matrix on screen
*/
void printConfusionMatrix(const std::vector<std::vector<int> >& confusionMatrix,
	const std::set<std::string>& classes) {
	for (auto it = classes.begin(); it != classes.end(); ++it) {
		std::cout << *it << " ";
	}
	std::cout << std::endl;
	for (size_t i = 0; i < confusionMatrix.size(); i++) {
		for (size_t j = 0; j < confusionMatrix[i].size(); j++) {
			std::cout << confusionMatrix[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

/**
* Get the accuracy for a model (i.e., percentage of correctly predicted
* test samples)
*/
float getAccuracy(const std::vector<std::vector<int> >& confusionMatrix) {
	int hits = 0;
	int total = 0;
	for (size_t i = 0; i < confusionMatrix.size(); i++) {
		for (size_t j = 0; j < confusionMatrix.at(i).size(); j++) {
			if (i == j) hits += confusionMatrix.at(i).at(j);
			total += confusionMatrix.at(i).at(j);
		}
	}
	return hits / (float)total;
}

/**
* Save our obtained models (neural network, bag of words vocabulary
* and class names) to use it later
*/
void saveModels(cv::Ptr<cv::ml::ANN_MLP> mlp, const cv::Mat& vocabulary,
	const std::set<std::string>& classes) {
	mlp->save("mlp.yaml");
	cv::FileStorage fs("vocabulary.yaml", cv::FileStorage::WRITE);
	fs << "vocabulary" << vocabulary;
	fs.release();
	std::ofstream classesOutput("classes.txt");
	for (auto it = classes.begin(); it != classes.end(); ++it) {
		classesOutput << getClassId(classes, *it) << "\t" << *it << std::endl;
	}
	classesOutput.close();
}

/**
* Get a confusion matrix from a set of test samples and their expected
* outputs, modified to take an int param for N x N.
*/
std::vector<std::vector<int> > getConfusionMatrix(cv::Ptr<cv::ml::ANN_MLP> mlp,
	const cv::Mat& testSamples, const std::vector<int>& testOutputExpected, const int n) {
	cv::Mat testOutput;
	mlp->predict(testSamples, testOutput);
	std::vector<std::vector<int> > confusionMatrix(n, std::vector<int>(n));
	for (int i = 0; i < testOutput.rows; i++) {
		int predictedClass = getPredictedClass(testOutput.row(i));
		int expectedClass = testOutputExpected.at(i);
		confusionMatrix[expectedClass][predictedClass]++;
	}
	return confusionMatrix;
}

/**
* Get a probability pair of a single (the first) ANN_MLP prediction result
*/
ProbabilityStruct getProbabilityPair(cv::Ptr<cv::ml::ANN_MLP> mlp,
	const cv::Mat& testSample, const std::set<std::string>& classes) {
	ProbabilityStruct retVal;

	// put the sample into the neural network
	cv::Mat testOutput;
	mlp->predict(testSample, testOutput, cv::ml::StatModel::Flags::RAW_OUTPUT);

	// determine the highest probability and class
	float maxPrediction = testOutput.at<float>(0);
	int maxPredictionIndex = 0;
	const float* ptrPredictions = testOutput.ptr<float>(0);
	auto classIterator = classes.begin()++;
	retVal.classname = *classIterator; // start with the assumption first is largest
	for (int i = 0; i < testOutput.cols; i++) {
		float prediction = *ptrPredictions++;
		if (prediction > maxPrediction) {
			maxPrediction = prediction;
			maxPredictionIndex = i;
			retVal.classname = *classIterator;
		}
		++classIterator;
	}

	// determine the next probability and class
	float nextPrediction;
	int nextPredictionIndex;

	if (maxPredictionIndex == 0) { // max was start, so go in reverese
		nextPrediction = testOutput.at<float>(testOutput.cols - 1);
		nextPredictionIndex = testOutput.cols - 1;
		ptrPredictions = testOutput.ptr<float>(testOutput.cols - 1);
		auto rClassIterator = classes.rbegin()++;
		retVal.nextClassname = *rClassIterator; // start with the assumption last is next largest
		for (int i = testOutput.cols - 1; i > 0; i--) {
			float prediction = *(--ptrPredictions);
			if (prediction > nextPrediction && prediction < maxPrediction) {
				nextPrediction = prediction;
				nextPredictionIndex = i;
				retVal.nextClassname = *rClassIterator;
			}
			rClassIterator++;
		}
	}
	else {
		nextPrediction = testOutput.at<float>(0);
		nextPredictionIndex = 0;
		ptrPredictions = testOutput.ptr<float>(0);
		classIterator = classes.begin()++;
		retVal.nextClassname = *classIterator; // start with the assumption first is next largest
		for (int i = 0; i < testOutput.cols; i++) {
			float prediction = *ptrPredictions++;
			if (prediction > nextPrediction && prediction < maxPrediction) {
				nextPrediction = prediction;
				nextPredictionIndex = i;
				retVal.nextClassname = *classIterator;
			}
			classIterator++;
		}
	}

	// save the probabilities
	retVal.probability = (maxPrediction + testOutput.cols) / (testOutput.cols * 2);
	retVal.nextProbability = (nextPrediction + testOutput.cols) / (testOutput.cols * 2);

	// std::cout << "maxPrediction " << maxPrediction << " testOutput.cols " << testOutput.cols << " (maxPrediction + testOutput.cols) " << (maxPrediction + testOutput.cols) << " (testOutput.cols * 2) " << (testOutput.cols * 2) << std::endl;
	// std::cout << "nextPrediction " << nextPrediction << " testOutput.cols " << testOutput.cols << " (nextPrediction + testOutput.cols) " << (nextPrediction + testOutput.cols) << " (testOutput.cols * 2) " << (testOutput.cols * 2) << std::endl;

	return retVal;
}

/**
* Display the image with color coded confidence %s to the user and optionally write the output to file in output directory.
*/
void displayConfidence(const cv::Mat &image, const ProbabilityStruct &thisResult, const std::set<std::string>& classes, const std::string &classname, const std::string &outputDir, const std::string &filename) {
	cv::Mat frame = image.clone();
	if (frame.type() == 0)
		cv::cvtColor(frame, frame, CV_GRAY2RGB); // convert to color to allow color text

	// color code the confidence text based on the probability and actual result
	cv::Scalar color;
	cv::Scalar color2;
	if (getClassId(classes, classname) == getClassId(classes, thisResult.classname)) { // correct primary
		color = cv::Scalar(0, thisResult.probability * 255, (1 - thisResult.probability) * 255); // r/y/g
		color2 = cv::Scalar(0, thisResult.nextProbability * 255, (1 - thisResult.nextProbability) * 255); // r/y/g
	}
	else { // incorrect primary
		color = cv::Scalar(0, 0, (1 - thisResult.probability) * 255); // r
		if (getClassId(classes, classname) == getClassId(classes, thisResult.nextClassname)) { // correct secondary
			color2 = cv::Scalar(0, thisResult.nextProbability * 255, (1 - thisResult.nextProbability) * 255); // r/y/g
		}
		else { // incorrect secondary
			color2 = cv::Scalar(0, 0, (1 - thisResult.nextProbability) * 255); // r
		}
	}

	// get classname and probability in a string
	std::string confidence = thisResult.classname + " " +
		boost::lexical_cast<std::string>(thisResult.probability * 100).substr(0, 5) + "%";
	std::string confidenceNext = thisResult.nextClassname + " " +
		boost::lexical_cast<std::string>(thisResult.nextProbability * 100).substr(0, 5) + "%";

	// put the text and display it
	putText(frame, confidence, cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 2, color, 2, cv::LINE_8, false);
	putText(frame, confidenceNext, cv::Point(10, 50), cv::FONT_HERSHEY_PLAIN, 1, color2, 2, cv::LINE_8, false);
	imshow("NN Testing Confidence", frame);

	// write image to file?
	if (outputDir != "") {
		_mkdir(outputDir.c_str());
		int start = filename.find_last_of('\\') + 1;
		imwrite(outputDir + "\\" + filename.substr(start, filename.length()), frame);
	}
}

//---------------------------- Main ----------------------------------
int main(int argc, char** argv) {
	if (argc != 3 && argc != 4 && argc != 5) {
		std::cerr << "Usage: " << argv[0] << " <IMG_DIR> <NETWORK_INPUT_LAYER_SIZE> ";
		std::cerr << "<TRAIN_SPLIT_RATIO_OPT> <OUTPUT_DIR_OPT>" << std::endl;
		exit(-1);
	}

	std::string imagesDir = argv[1];
	int networkInputSize = atoi(argv[2]);
	bool isTrain = false;
	float trainSplitRatio = 0.0f;
	std::string outputDir = "";

	if (argc >= 4) {
		trainSplitRatio = atof(argv[3]);

		if (trainSplitRatio < 0 || trainSplitRatio > 1) {
			std::cerr << "TRAIN_SPLIT_RATIO must be > 0 and <= 1 for training. ";
			std::cerr << "Omit or set to 0 for testing only." << std::endl;
			exit(-1);
		}

		if (trainSplitRatio != 0) isTrain = true;
	}

	if (argc >= 5) outputDir = argv[4];

#ifdef HAVE_TBB
	std::cout << "Running with TBB" << std::endl;
#else
#ifdef _OPENMP
	std::cout << "Running with OpenMP" << std::endl;
#else
	std::cout << "Running without OpenMP and without TBB" << std::endl;
#endif
#endif

	if (isTrain) {
		std::cout << "Reading training set..." << std::endl;
		double start = (double)cv::getTickCount();
		std::vector<std::string> files = getFilesInDirectory(imagesDir);
		std::random_shuffle(files.begin(), files.end());

		cv::Mat descriptorsSet;
		std::vector<ImageData*> descriptorsMetadata;
		std::set<std::string> classes;
#ifdef USE_DPM_TO_TRAIN
		readImagesWithDPM(files.begin(), files.begin() + (size_t)(files.size() * trainSplitRatio),
			[&](const std::string& classname, const cv::Mat& descriptors, const cv::Mat& image, const std::string& filename) {
#else
		readImages(files.begin(), files.begin() + (size_t)(files.size() * trainSplitRatio),
			[&](const std::string& classname, const cv::Mat& descriptors, const cv::Mat& image, const std::string& filename) {
#endif // USE_DPM_TO_TRAIN
			// Append to the set of classes
			classes.insert(classname);
			// Append to the list of descriptors
			descriptorsSet.push_back(descriptors);
			// Append metadata to each extracted feature
			ImageData* data = new ImageData;
			data->classname = classname;
			data->bowFeatures = cv::Mat::zeros(cv::Size(networkInputSize, 1), CV_32F);
			for (int j = 0; j < descriptors.rows; j++) {
				descriptorsMetadata.push_back(data);
			}
#ifdef DISPLAY_CONFIDENCE 
			imshow("NN Training Image", image);
			int c = cv::waitKey(WAITKEY_DELAY);
			if ((char)c == 27) exit(0); // escape
#endif // DISPLAY_CONFIDENCE
		});
		std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

		std::cout << "Creating vocabulary..." << std::endl;
		start = (double)cv::getTickCount();
		cv::Mat labels;
		cv::Mat vocabulary;
		// Use k-means to find k centroids (the words of our vocabulary)
		cv::kmeans(descriptorsSet, networkInputSize, labels, cv::TermCriteria(cv::TermCriteria::EPS +
			cv::TermCriteria::MAX_ITER, 10, 0.01), 1, cv::KMEANS_PP_CENTERS, vocabulary);
		// No need to keep it on memory anymore
		descriptorsSet.release();
		std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

		// Convert a set of local features for each image in a single descriptors
		// using the bag of words technique
		std::cout << "Getting histograms of visual words..." << std::endl;
		int* ptrLabels = (int*)(labels.data);
		int size = labels.rows * labels.cols;
		for (int i = 0; i < size; i++) {
			int label = *ptrLabels++;
			ImageData* data = descriptorsMetadata[i];
			data->bowFeatures.at<float>(label)++;
		}

		// Filling matrixes to be used by the neural network
		std::cout << "Preparing neural network..." << std::endl;
		cv::Mat trainSamples;
		cv::Mat trainResponses;
		std::set<ImageData*> uniqueMetadata(descriptorsMetadata.begin(), descriptorsMetadata.end());
		for (auto it = uniqueMetadata.begin(); it != uniqueMetadata.end(); ) {
			ImageData* data = *it;
			cv::Mat normalizedHist;
			cv::normalize(data->bowFeatures, normalizedHist, 0, data->bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
			trainSamples.push_back(normalizedHist);
			trainResponses.push_back(getClassCode(classes, data->classname));
			delete *it; // clear memory
			it++;
		}
		descriptorsMetadata.clear();

		// Training neural network
		std::cout << "Training neural network..." << std::endl;
		start = cv::getTickCount();
		cv::Ptr<cv::ml::ANN_MLP> mlp = getTrainedNeuralNetwork(trainSamples, trainResponses);
		std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

		// We can clear memory now 
		trainSamples.release();
		trainResponses.release();

		// Train FLANN 
		std::cout << "Training FLANN..." << std::endl;
		start = cv::getTickCount();
		cv::FlannBasedMatcher flann;
		flann.add(vocabulary);
		flann.train();
		std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

		// skip testing if no images to test
		if (trainSplitRatio < 1) {
			// Reading test set 
			std::cout << "Reading test set..." << std::endl;
			start = cv::getTickCount();
			cv::Mat testSamples;
			std::vector<int> testOutputExpected;
#ifdef USE_DPM_TO_TEST
			readImagesWithDPM(files.begin() + (size_t)(files.size() * trainSplitRatio), files.end(),
				[&](const std::string& classname, const cv::Mat& descriptors, const cv::Mat& image, const std::string& filename) {
#else
			readImages(files.begin() + (size_t)(files.size() * trainSplitRatio), files.end(),
				[&](const std::string& classname, const cv::Mat& descriptors, const cv::Mat& image, const std::string& filename) {
#endif // USE_DPM_TO_TEST
				// Get histogram of visual words using bag of words technique
				cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, networkInputSize);
				cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
				testSamples.push_back(bowFeatures);
				testOutputExpected.push_back(getClassId(classes, classname));

				// cv::Mat rawOutput;
				// mlp->predict(bowFeatures, rawOutput, cv::ml::StatModel::Flags::RAW_OUTPUT);
				// std::cout << rawOutput << std::endl;

				ProbabilityStruct thisResult = getProbabilityPair(mlp, bowFeatures, classes);
				std::cout << thisResult.classname << " " << (thisResult.probability * 100) << "% ";
				std::cout << thisResult.nextClassname << " " << (thisResult.nextProbability * 100) << "%" << std::endl;

#ifdef DISPLAY_CONFIDENCE 
				displayConfidence(image, thisResult, classes, classname, outputDir, filename);
				int c = cv::waitKey(WAITKEY_DELAY);
				if ((char)c == 27) exit(0); // escape
#endif // DISPLAY_CONFIDENCE

			});
			std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

			// Get confusion matrix of the test set
			std::vector<std::vector<int> > confusionMatrix = getConfusionMatrix(mlp, testSamples, testOutputExpected, classes.size());

			// Get accuracy of our model
			std::cout << "Confusion matrix: " << std::endl;
			printConfusionMatrix(confusionMatrix, classes);
			std::cout << "Accuracy: " << getAccuracy(confusionMatrix) << std::endl;
		}

		// Save models
		std::cout << "Saving models..." << std::endl;
		saveModels(mlp, vocabulary, classes);
	}
	else { // is test
		std::cout << "IN TEST MODE ONLY" << std::endl << "Reading data..." << std::endl;
		double start = (double)cv::getTickCount();

		// read in the mlp yaml file
		std::ifstream test("mlp.yaml");
		if (!test.good()) {
			std::cerr << "mlp.yaml not found. Run in train mode first." << std::endl;
			exit(-1);
		}
		test.close();
		cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::load("mlp.yaml");

		// read in the vocabulary file into the flann
		cv::FlannBasedMatcher flann;
		cv::FileStorage fsVOCAB("vocabulary.yaml", cv::FileStorage::READ);
		if (!fsVOCAB.isOpened()) {
			std::cerr << "vocabulary.yaml not found. Run in train mode first." << std::endl;
			exit(-1);
		}
		cv::Mat vocabs;
		fsVOCAB["vocabulary"] >> vocabs;
		fsVOCAB.release();
		flann.add(vocabs);
		flann.train();

		// Read in classes
		std::set<std::string> classes;
		std::ifstream classesInput("classes.txt");
		if (!classesInput.is_open()) {
			std::cerr << "classes.txt not found. Run in train mode first." << std::endl;
			exit(-1);
		}
		while (classesInput.good()) {
			int test = classesInput.peek();
			if (!std::isdigit(test)) break;
			classesInput >> test; // trash the id
			std::string classname;
			classesInput >> classname;
			classes.insert(classname);
			test = classesInput.get(); // get rid of new line
		}
		classesInput.close();		

		std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

		// Reading test set 
		std::cout << "Reading test set..." << std::endl;
		start = cv::getTickCount();
		cv::Mat testSamples;
		std::vector<int> testOutputExpected;
		std::vector<std::string> files = getFilesInDirectory(imagesDir);
		cv::Mat frame;
#ifdef USE_DPM_TO_TEST
		readImagesWithDPM(files.begin(), files.end(),
			[&](const std::string& classname, const cv::Mat& descriptors, const cv::Mat& image, const std::string& filename) {
#else
		readImages(files.begin(), files.end(),
			[&](const std::string& classname, const cv::Mat& descriptors, const cv::Mat& image, const std::string& filename) {
#endif // USE_DPM_TO_TEST
			// Get histogram of visual words using bag of words technique
			cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, networkInputSize);
			cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
			testSamples.push_back(bowFeatures);
			testOutputExpected.push_back(getClassId(classes, classname));

			// cv::Mat rawOutput;
			// mlp->predict(bowFeatures, rawOutput, cv::ml::StatModel::Flags::RAW_OUTPUT);
			// std::cout << rawOutput << std::endl;

			ProbabilityStruct thisResult = getProbabilityPair(mlp, bowFeatures, classes);
			std::cout << thisResult.classname << " " << (thisResult.probability * 100) << "% ";
			std::cout << thisResult.nextClassname << " " << (thisResult.nextProbability * 100) << "%" << std::endl;

#ifdef DISPLAY_CONFIDENCE 
			displayConfidence(image, thisResult, classes, classname, outputDir, filename);
			int c = cv::waitKey(WAITKEY_DELAY);
			if ((char)c == 27) exit(0); // escape
#endif // DISPLAY_CONFIDENCE
		});
		std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

		// Get confusion matrix of the test set
		std::vector<std::vector<int> > confusionMatrix = getConfusionMatrix(mlp, testSamples, testOutputExpected, classes.size());
		
		// Get accuracy of our model
		std::cout << "Confusion matrix: " << std::endl;
		printConfusionMatrix(confusionMatrix, classes);
		std::cout << "Accuracy: " << getAccuracy(confusionMatrix) << std::endl;
	}

	return 0;
}