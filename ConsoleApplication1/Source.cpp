// load and display an image
/*
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	String imageName("pic.png"); // by default
	if (argc > 1)
	{
		imageName = argv[1];
	}
	Mat image;
	image = imread(imageName, IMREAD_COLOR); // Read the file
	if (image.empty())                      // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", image);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}
*/

//loading an image and changing it to grayscale then save
/*
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	String imageName("pic.png"); // by default
	if (argc > 1)
	{
		imageName = argv[1];
	}
	Mat image;
	image = imread(imageName, IMREAD_COLOR); // Read the file
	if (image.empty())                      // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	
	Mat gray_image;
	cvtColor(image, gray_image, COLOR_BGR2GRAY);
	imwrite("Gray_Image.jpg", gray_image);
	namedWindow(imageName, WINDOW_AUTOSIZE);
	namedWindow("Gray image", WINDOW_AUTOSIZE);
	imshow(imageName, image);
	imshow("Gray image", gray_image);
	waitKey(0);
	return 0;
}
*/

// searching image for smaller image

/*
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace std;
using namespace cv;
bool use_mask;
Mat img; Mat templ; Mat mask; Mat result;
const char* image_window = "Source Image";
const char* result_window = "Result window";
int match_method;
int max_Trackbar = 5;
void MatchingMethod(int, void*);
int main(int argc, char** argv)
{
	argv[1] = "eric5.jpg"; // the whole image
	argv[2] = "eric_face.jpg";       // what you're looking for in the image
	//if (argc < 3)
	//{
	//	cout << "Not enough parameters" << endl;
	//	cout << "Usage:\n./MatchTemplate_Demo <image_name> <template_name> [<mask_name>]" << endl;
	//	return -1;
	//}
	img = imread(argv[1], IMREAD_COLOR);
	templ = imread(argv[2], IMREAD_COLOR);
	if (argc > 3) {
		use_mask = true;
		mask = imread(argv[3], IMREAD_COLOR);
	}
	if (img.empty() || templ.empty() || (use_mask && mask.empty()))
	{
		cout << "Can't read one of the images" << endl;
		return -1;
	}
	namedWindow(image_window, WINDOW_AUTOSIZE);
	namedWindow(result_window, WINDOW_AUTOSIZE);
	const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
	createTrackbar(trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod);
	MatchingMethod(0, 0);
	waitKey(0);
	return 0;
}
void MatchingMethod(int, void*)
{
	Mat img_display;
	img.copyTo(img_display);
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	result.create(result_rows, result_cols, CV_32FC1);
	bool method_accepts_mask = (CV_TM_SQDIFF == match_method || match_method == CV_TM_CCORR_NORMED);
	if (use_mask && method_accepts_mask)
	{
		matchTemplate(img, templ, result, match_method, mask);
	}
	else
	{
		matchTemplate(img, templ, result, match_method);
	}
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}
	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
	rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
	imshow(image_window, img_display);
	imshow(result_window, result);
	return;
}
*/

//open webcam and find face

//#include "opencv2/objdetect.hpp"
//#include "opencv2/videoio.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
//#include <string>
//#include <iostream>
//#include <stdio.h>
//
//using namespace std;
//using namespace cv;
//
///** Function Headers */
//void detectAndDisplay(Mat frame);
//
///** Global variables */
//String face_cascade_name, eyes_cascade_name;
//CascadeClassifier face_cascade;
//CascadeClassifier eyes_cascade;
//String window_name = "Capture - Face detection";
//
///** @function main */
//int main(int argc, const char** argv)
//{
//	CommandLineParser parser(argc, argv,
//		"{help h||}"
//		"{face_cascade|haarcascade_frontalface_alt.xml|}"
//		"{eyes_cascade|haarcascade_eye_tree_eyeglasses.xml|}");
//
//	cout << "\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
//		"You can use Haar or LBP features.\n\n";
//	parser.printMessage();
//
//	face_cascade_name = parser.get<string>("face_cascade");
//	eyes_cascade_name = parser.get<string>("eyes_cascade");
//	VideoCapture capture;
//	Mat frame;
//
//	//-- 1. Load the cascades
//	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); return -1; };
//	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading eyes cascade\n"); return -1; };
//
//	//-- 2. Read the video stream
//	capture.open(0);
//	if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }
//
//	while (capture.read(frame))
//	{
//		if (frame.empty())
//		{
//			printf(" --(!) No captured frame -- Break!");
//			break;
//		}
//
//		//-- 3. Apply the classifier to the frame
//		detectAndDisplay(frame);
//
//		char c = (char)waitKey(10);
//		if (c == 27) { break; } // escape
//	}
//	return 0;
//}
//
///** @function detectAndDisplay */
//void detectAndDisplay(Mat frame)
//{
//	std::vector<Rect> faces;
//	Mat frame_gray;
//
//	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
//	equalizeHist(frame_gray, frame_gray);
//
//	//-- Detect faces
//	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
//
//	for (size_t i = 0; i < faces.size(); i++)
//	{
//		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
//		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
//
//		Mat faceROI = frame_gray(faces[i]);
//		std::vector<Rect> eyes;
//
//		//-- In each face, detect eyes
//		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(10, 30));
//
//		for (size_t j = 0; j < eyes.size(); j++)
//		{
//			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
//			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
//			circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
//		}
//	}
//	//-- Show what you got
//	imshow(window_name, frame);
//}


// canny edge detection
/*
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/// Global variables

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";


//* @function CannyThreshold
//* @brief Trackbar callback - Canny thresholds input with a ratio 1:3
//
//void CannyThreshold(int, void*)
//{
//	/// Reduce noise with a kernel 3x3
//	blur(src_gray, detected_edges, Size(3, 3));
//
//	/// Canny detector
//	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
//
//	/// Using Canny's output as a mask, we display our result
//	dst = Scalar::all(0);
//
//	src.copyTo(dst, detected_edges);
//	imshow(window_name, dst);
//}


// @function main 
int main(int argc, char** argv)
{
	argv[1] = "eric1.jpg";
	/// Load an image
	src = imread(argv[1]);

	if (!src.data)
	{
		return -1;
	}

	/// Create a matrix of the same type and size as src (for dst)
	dst.create(src.size(), src.type());

	/// Convert the image to grayscale
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Create a window
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	/// Create a Trackbar for user to enter threshold
	createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);

	/// Show the image
	CannyThreshold(0, 0);

	/// Wait until user exit program by pressing a key
	waitKey(0);

	return 0;
}
*/

//testing stuff
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String face_cascade_name;
CascadeClassifier face_cascade;
String window_name = "Face Detection";
bool use_mask;
Mat img; Mat templ; Mat mask; Mat result;
const char* result_window = "Result window";
int match_method;
int max_Trackbar = 5;
const char* image_window = "Image window";

void MatchingMethod(int, void*);

/** @function main */
int main(int argc, const char** argv)
{
	CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|haarcascade_frontalface_alt.xml|}");

	argv[1] = "eric1.jpg";	     // the whole image
	String imageName(argv[1]);		 // whole image

	face_cascade_name = parser.get<string>("face_cascade");
	VideoCapture capture;
	Mat frame;

	Mat image;
	namedWindow(image_window, WINDOW_AUTOSIZE); // Create a window for display.

	Mat faceImage;
	//-- 2. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); return -1; };

	//-- 3. Read the video stream
	capture.open(0);
	if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }

	while (capture.read(frame))
	{
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}
		
		//-- 4. Show the group image that we're going to search in
		image = imread(imageName, IMREAD_COLOR); // Read the file

		//-- 5. Apply the classifier to the frame to video
		detectAndDisplay(frame);

		// show the image of face cut out from video
		argv[2] = "new_image.jpg";       // what you're looking for in the image
		String compareImage(argv[2]);	 // what you're looking for

		faceImage = imread(compareImage, IMREAD_COLOR);
		imshow("Face", faceImage);

		//-- 6. Search image for face
		img = imread(argv[1], IMREAD_COLOR);
		templ = imread(argv[2], IMREAD_COLOR);

		//namedWindow(image_window, WINDOW_AUTOSIZE);
		const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
		createTrackbar(trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod);
		MatchingMethod(0, 0);

		char c = (char)waitKey(10);
		if (c == 27) { break; } // escape
	}
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);

		// Take a picture of face and save image
		Mat faceROI = frame(faces[i]);
		imwrite("new_image.jpg", faceROI);
		faceROI = frame_gray(faces[i]);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
//		Mat faceROI = frame_gray(faces[i]);
//		imwrite("new_image.jpg", faceROI);
//		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
	}

	//-- Show video
	imshow(window_name, frame);
}

void MatchingMethod(int, void*)
{
	Mat img_display;
	img.copyTo(img_display);
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	result.create(result_rows, result_cols, CV_32FC1);
	bool method_accepts_mask = (CV_TM_SQDIFF == match_method || match_method == CV_TM_CCORR_NORMED);
	if (use_mask && method_accepts_mask)
	{
		matchTemplate(img, templ, result, match_method, mask);
	}
	else
	{
		matchTemplate(img, templ, result, match_method);
	}
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}
	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
	rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
	imshow(image_window, img_display);
	return;
}
