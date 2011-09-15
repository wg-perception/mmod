// My Multi-Modality experimental functions.
//
// console application.
// Gary Bradski Aug 15, 2011.
// 
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

#include "mmod_general.h"
#include "mmod_objects.h"
#include "mmod_mode.h"
#include "mmod_features.h"

//For serialization
#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>


using namespace cv;
using namespace std;

void help() {
	cout << "My Multi-Mod Experiments. Call convention:\n\n"
		"  MyMMod image_list\n\n"
		"Where: image_list is space separated list of path/filename of images\n"
		"       \n"
		"Hit 'p' to pause/unpause, ESC to quit.\n" << endl;
}

//Just debug image print out
template<typename T>
void prnMatC1(const Mat &in) {
	for (int y = 0; y < 5; ++y) {
		const T* o = in.ptr<T> (y);
		for (int x = 0; x < 5; ++x, ++o) {
			cout << (int) (*o) << ", ";
		}
		cout << endl;
	}
	cout << "\n\n" << endl;
}

///////////////////////////////////////////////////////////////////////////////////////////
// FEATURE COMPUTATION
///////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////
//TEST FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////
void coutuchar(uchar c) {
	for (int i = 0; i < 8; ++i) {
		uchar foo = 128 >> i;
		if (foo & c)
			cout << "1 ";
		else
			cout << "0 ";
	}
	cout << endl;
}
void reportfillCosDistTest(uchar model, uchar image, mmod_general &mm) {
	cout << "Model: ";
	coutuchar(model);
	cout << "Image: ";
	coutuchar(image);
	cout << "Match: " << mm.match(model, image) << "\n" << endl;
}
void fillCosDistTest() {
	mmod_general foo;
	foo.fillCosDist();
	cout << "This is the cos distance function:\n"
			<< "1.000000000,//(cos(0)+1)/2      0 (dist in bits)\n"
				"0.961939766,//(cos(22.5)+1)/2   1\n"
				"0.853553391,//(cos(45)+1)/2     2\n"
				"0.691341716,//(cos(67.5)+1)/2   3\n"
				"0.500000000,//(cos(90)+1)/2     4\n"
				"0.308658284,//(cos(112.5)+1)/2  5\n"
				"0.146446609,//(cos(135)+1)/2    6\n"
				"0.038060234,//(cos(157.5)+1)/2  7\n"
				"0.000000000//(cos(180)+1)/2     8\n" << endl;
	uchar model, image;
	model = 0;
	image = 0;
	reportfillCosDistTest(model, image, foo);
	model = 0;
	image = 12;
	reportfillCosDistTest(model, image, foo);
	model = 64;
	image = 0;
	reportfillCosDistTest(model, image, foo);

	model = 0;
	image = 3;
	reportfillCosDistTest(model, image, foo);
	model = 0;
	image = 33;
	reportfillCosDistTest(model, image, foo);
	model = 0;
	image = 145;
	reportfillCosDistTest(model, image, foo);
	model = 0;
	image = 205;
	reportfillCosDistTest(model, image, foo);
	model = 0;
	image = 244;
	reportfillCosDistTest(model, image, foo);

	model = 1;
	image = 3;
	reportfillCosDistTest(model, image, foo);
	model = 1;
	image = 33;
	reportfillCosDistTest(model, image, foo);
	model = 1;
	image = 145;
	reportfillCosDistTest(model, image, foo);
	model = 1;
	image = 205;
	reportfillCosDistTest(model, image, foo);
	model = 1;
	image = 244;
	reportfillCosDistTest(model, image, foo);

	model = 2;
	image = 3;
	reportfillCosDistTest(model, image, foo);
	model = 2;
	image = 33;
	reportfillCosDistTest(model, image, foo);
	model = 2;
	image = 145;
	reportfillCosDistTest(model, image, foo);
	model = 2;
	image = 205;
	reportfillCosDistTest(model, image, foo);
	model = 2;
	image = 244;
	reportfillCosDistTest(model, image, foo);

	model = 4;
	image = 3;
	reportfillCosDistTest(model, image, foo);
	model = 4;
	image = 33;
	reportfillCosDistTest(model, image, foo);
	model = 4;
	image = 145;
	reportfillCosDistTest(model, image, foo);
	model = 4;
	image = 205;
	reportfillCosDistTest(model, image, foo);
	model = 4;
	image = 244;
	reportfillCosDistTest(model, image, foo);

	model = 8;
	image = 3;
	reportfillCosDistTest(model, image, foo);
	model = 8;
	image = 33;
	reportfillCosDistTest(model, image, foo);
	model = 8;
	image = 145;
	reportfillCosDistTest(model, image, foo);
	model = 8;
	image = 205;
	reportfillCosDistTest(model, image, foo);
	model = 8;
	image = 244;
	reportfillCosDistTest(model, image, foo);

	model = 16;
	image = 3;
	reportfillCosDistTest(model, image, foo);
	model = 16;
	image = 33;
	reportfillCosDistTest(model, image, foo);
	model = 16;
	image = 145;
	reportfillCosDistTest(model, image, foo);
	model = 16;
	image = 205;
	reportfillCosDistTest(model, image, foo);
	model = 16;
	image = 244;
	reportfillCosDistTest(model, image, foo);

	model = 32;
	image = 3;
	reportfillCosDistTest(model, image, foo);
	model = 32;
	image = 33;
	reportfillCosDistTest(model, image, foo);
	model = 32;
	image = 145;
	reportfillCosDistTest(model, image, foo);
	model = 32;
	image = 205;
	reportfillCosDistTest(model, image, foo);
	model = 32;
	image = 244;
	reportfillCosDistTest(model, image, foo);

	model = 64;
	image = 3;
	reportfillCosDistTest(model, image, foo);
	model = 64;
	image = 33;
	reportfillCosDistTest(model, image, foo);
	model = 64;
	image = 145;
	reportfillCosDistTest(model, image, foo);
	model = 64;
	image = 205;
	reportfillCosDistTest(model, image, foo);
	model = 64;
	image = 244;
	reportfillCosDistTest(model, image, foo);

	model = 128;
	image = 3;
	reportfillCosDistTest(model, image, foo);
	model = 128;
	image = 33;
	reportfillCosDistTest(model, image, foo);
	model = 128;
	image = 145;
	reportfillCosDistTest(model, image, foo);
	model = 128;
	image = 205;
	reportfillCosDistTest(model, image, foo);
	model = 128;
	image = 244;
	reportfillCosDistTest(model, image, foo);

	model = 156;
	image = 205;
	reportfillCosDistTest(model, image, foo);
	model = 156;
	image = 0;
	reportfillCosDistTest(model, image, foo);
}

void mmod_general_test() {
	/*
	 * Input
	 *  1  64   2   0    0
	 * 64   1   2   0    0
	 *128  64   0   8    4
	 *  8   4   8  16  128
	 *  2   4   4  32   64
	 */
	//create a test image
	Mat in(5, 5, CV_8U);
	in.at<uchar> (0, 0) = 1;
	in.at<uchar> (0, 1) = 64;
	in.at<uchar> (0, 2) = 2;
	in.at<uchar> (0, 3) = 0;
	in.at<uchar> (0, 4) = 0;

	in.at<uchar> (1, 0) = 64;
	in.at<uchar> (1, 1) = 1;
	in.at<uchar> (1, 2) = 2;
	in.at<uchar> (1, 3) = 0;
	in.at<uchar> (1, 4) = 0;

	in.at<uchar> (2, 0) = 128;
	in.at<uchar> (2, 1) = 64;
	in.at<uchar> (2, 2) = 0;
	in.at<uchar> (2, 3) = 8;
	in.at<uchar> (2, 4) = 4;

	in.at<uchar> (3, 0) = 8;
	in.at<uchar> (3, 1) = 4;
	in.at<uchar> (3, 2) = 8;
	in.at<uchar> (3, 3) = 16;
	in.at<uchar> (3, 4) = 128;

	in.at<uchar> (4, 0) = 2;
	in.at<uchar> (4, 1) = 4;
	in.at<uchar> (4, 2) = 4;
	in.at<uchar> (4, 3) = 32;
	in.at<uchar> (4, 4) = 64;
	Mat out;
	cout << "Test: mmod_general::SumAroundEachPixel8UC1 input matrix:\n"
			<< endl;
	for (int y = 0; y < 5; ++y) {
		uchar *o = in.ptr<uchar> (y);
		for (int x = 0; x < 5; ++x, ++o) {
			cout << (int) (*o) << ", ";
		}
		cout << endl;
	}
	cout << "\n\n" << endl;
	mmod_general foo;
	foo.SumAroundEachPixel8UC1(in, out, 3, 1);
	cout << "Max Test: mmod_general::SumAroundEachPixel8UC1 output matrix:\n"
			<< endl;
	for (int y = 0; y < 5; ++y) {
		uchar *o = out.ptr<uchar> (y);
		for (int x = 0; x < 5; ++x, ++o) {
			cout << (int) (*o) << ", ";
		}
		cout << endl;
	}
	cout << "\n\n" << endl;
	cout << "/*\n"
		"* Max output should be\n"
		"*  1   1   2   2    0\n"
		"* 64  64   2   2    4\n"
		"* 64   8   8   8    4\n"
		"*  4   4   4   4  128\n"
		"*  4   4   4  32   64\n"
		"*/\n" << endl;
	foo.SumAroundEachPixel8UC1(in, in, 3, 0);
	cout << "OR Test: mmod_general::SumAroundEachPixel8UC1 output matrix:\n"
			<< endl;
	for (int y = 0; y < 5; ++y) {
		uchar *o = in.ptr<uchar> (y);
		for (int x = 0; x < 5; ++x, ++o) {
			cout << (int) (*o) << ", ";
		}
		cout << endl;
	}
	cout << "\n/*\n"
		"* OR output should be\n"
		"*  65,  67,  67,   2,   0,\n"
		"* 193, 195,  75,  14,  12,\n"
		"* 205, 207,  95, 158, 156,\n"
		"* 206, 206, 124, 252, 252,\n"
		"*  14,  14,  60, 252, 240,\n"
		"*/ \n";
	cout << "\n\n" << endl;

}

void prnR(const Rect R)
{ cout << "R(x,y,w,h) = ("<< R.x << ", " << R.y << ", " << R.width << ", " << R.height << ")" << endl;}

void test_match_a_patch_n_display_feature() {
	mmod_features f;
	mmod_general g;

	f.session_ID = "testID";
	f.object_ID = "test";
	f.frame_number.resize(3);
	f.features.resize(3);
	f.offsets.resize(3);

	int i = 0;
	f.frame_number.push_back(i);
	f.features[i].push_back(128);
	f.features[i].push_back(64);
	f.features[i].push_back(32);
	f.features[i].push_back(128);
	f.features[i].push_back(64);
	f.features[i].push_back(128);
	f.offsets[i].push_back(Point(-15, -30));
	f.offsets[i].push_back(Point(5, -10));
	f.offsets[i].push_back(Point(10, 15));
	f.offsets[i].push_back(Point(-12, 22));
	f.offsets[i].push_back(Point(0, 0));
	f.offsets[i].push_back(Point(5, 5));

	i = 1;
	f.frame_number.push_back(i);
	f.features[i].push_back(128);
	f.features[i].push_back(64);
	f.features[i].push_back(32);
	f.features[i].push_back(64);
	f.features[i].push_back(32);
	f.features[i].push_back(128);
	f.features[i].push_back(64);
	f.features[i].push_back(128);
	f.offsets[i].push_back(Point(-15, -30));
	f.offsets[i].push_back(Point(5, -10));
	f.offsets[i].push_back(Point(10, 15));
	f.offsets[i].push_back(Point(19, 30));
	f.offsets[i].push_back(Point(10, 10));
	f.offsets[i].push_back(Point(14, -12));
	f.offsets[i].push_back(Point(0, 0));
	f.offsets[i].push_back(Point(-15, 15));

	i = 2;
	f.frame_number.push_back(i);
	f.features[i].push_back(128);
	f.features[i].push_back(64);
	f.features[i].push_back(32);
	f.features[i].push_back(128);
	f.features[i].push_back(128);
	f.features[i].push_back(128);
	f.features[i].push_back(128);
	f.features[i].push_back(128);
	f.features[i].push_back(128);
	f.features[i].push_back(64);
	f.features[i].push_back(64);
	f.features[i].push_back(64);
	f.features[i].push_back(64);
	f.features[i].push_back(64);
	f.features[i].push_back(64);
	f.offsets[i].push_back(Point(-15, -30));
	f.offsets[i].push_back(Point(5, -10));
	f.offsets[i].push_back(Point(10, 15));
	f.offsets[i].push_back(Point(17, 17));
	f.offsets[i].push_back(Point(22, 22));
	f.offsets[i].push_back(Point(29, 29));
	f.offsets[i].push_back(Point(-17, 17));
	f.offsets[i].push_back(Point(-22, 22));
	f.offsets[i].push_back(Point(-29, 29));
	f.offsets[i].push_back(Point(-17, -17));
	f.offsets[i].push_back(Point(-22, -22));
	f.offsets[i].push_back(Point(-29, -29));
	f.offsets[i].push_back(Point(17, -17));
	f.offsets[i].push_back(Point(22, -22));
	f.offsets[i].push_back(Point(29, -29));

	f.bbox.push_back(Rect(-31, -16, 62, 32));
	f.bbox.push_back(Rect(-31, -20, 62, 40));
	f.bbox.push_back(Rect(-31, -31, 62, 62));

	//TEST RECTANGLE BOUNDS FINDING:
	cout << "Testing rectangle bounds finding" << endl;
	Rect R = f.find_max_template_size();

	Mat I = Mat::zeros(R.height, R.width, CV_8UC1);
	namedWindow("I", 0);
	i = 0;
	g.display_feature(I, f.features[i], f.offsets[i], f.bbox[i]);
	imshow("I", I);
	waitKey(0);
	i = 1;
	g.display_feature(I, f.features[i], f.offsets[i], f.bbox[i]);
	imshow("I", I);
	waitKey(0);
	i = 2;
	g.display_feature(I, f.features[i], f.offsets[i], f.bbox[i]);
	imshow("I", I);
	waitKey(0);

	cout << "TESTING match_a_patch:" << endl;
	float score;
	int index;
	i = 0;

	g.display_feature(I, f.features[i], f.offsets[i], f.bbox[i]);
	Point pp = Point(I.cols / 2, I.rows / 2);
	score = g.match_a_patch_bruteforce(I, pp, f, index);
	cout << "For feature " << i << " we matched it at " << index
			<< " with a score of " << score << endl;
	i = 1;
	g.display_feature(I, f.features[i], f.offsets[i], f.bbox[i]);
	score = g.match_a_patch_bruteforce(I, pp, f, index);
	cout << "For feature " << i << " we matched it at " << index
			<< " with a score of " << score << endl;
	i = 2;
	g.display_feature(I, f.features[i], f.offsets[i], f.bbox[i]);
	score = g.match_a_patch_bruteforce(I, pp, f, index);
	cout << "For feature " << i << " we matched it at " << index
			<< " with a score of " << score << endl;
}

////////////////////////////////////////////////
int test_object_rec() {
	cout << "\n\n============TEST_OBJECT_REC============\n" << endl;
	mmod_general g;
	vector < Mat > If;

//	RGB   If[] 0,1,2
//	Color If[] 3,4,5,
//	Depth If[] 6,7,8

	//Make ups some dummy images
	for (int i = 0; i < 9; ++i) {
		If.push_back(Mat::zeros(15,11, CV_8UC1));
	}
	//Make Mask
	Mat Mask = Mat::zeros(15,11, CV_8UC1); //Mask is 7x8 (x,y) around center (5,7)
	for (int y = 0; y < 15; ++y) {
		if ((y > 2) && (y < 11)) {
			uchar *m = Mask.ptr<uchar> (y);
			for (int x = 0; x < 11; ++x, ++m) {
				if ((x > 1) && (x < 9)) {
					*m = 1;
				}
			}
		}
	}

	Rect R(2,3,7,8);
	Rect roi(-11/2,-15/2,11,15);
	Rect roi2(-10,-10,20,20); //Just make a bigger feature
	//Make dummy image features
	vector<uchar> f;
	vector<Point> p;
	//Grad
	//0
	f.push_back(128);
	p.push_back(Point(-3,-5));
	f.push_back(128);
	p.push_back(Point(4,2));
	f.push_back(64);
	p.push_back(Point(0,0));//same
	f.push_back(64);
	p.push_back(Point(-3,-4));
	f.push_back(64);
	p.push_back(Point(3,3));
	g.display_feature(If[0],f,p,roi);
	f.clear(); p.clear();
	//1 Shift upper left point one to the right (Cos Match = 0, Total match = 0.6666 comp. with //0
	f.push_back(128);
	p.push_back(Point(-3,-5));
	f.push_back(128);
	p.push_back(Point(4,2));
	f.push_back(64);
	p.push_back(Point(0,0));//same
	f.push_back(64);
	p.push_back(Point(-3,-4));
	f.push_back(64);
	p.push_back(Point(3,3));
	f.push_back(128);
	p.push_back(Point(6,6));
	cout << "display feature" << endl;
	If[1] = Mat::zeros(30,30, CV_8UC1);
	g.display_feature(If[1],f,p,roi2);
	cout << "displayed" << endl;
	f.clear(); p.clear();
	//2 Upper left point changes from 64 to 4 (Cos Match = 0.5, Total Match 0.83333) comp. with //0
	f.push_back(128);
	p.push_back(Point(-3,-5));
	f.push_back(128);
	p.push_back(Point(4,2));
	f.push_back(64);
	p.push_back(Point(0,0));//same
	f.push_back(4);
	p.push_back(Point(-3,-4));
	f.push_back(64);
	p.push_back(Point(3,3));
	g.display_feature(If[2],f,p,roi);
	f.clear(); p.clear();
	//Color
	//3
	f.push_back(128);
	p.push_back(Point(-3,-5));
	f.push_back(128);
	p.push_back(Point(4,2));
	f.push_back(64);
	p.push_back(Point(0,0));//same
	f.push_back(64);
	p.push_back(Point(3,-4));
	f.push_back(64);
	p.push_back(Point(-3,3));
	g.display_feature(If[3],f,p,roi);
	f.clear(); p.clear();
	//4 Center point changes from 64 to 128 (Cos match = 0.961, Total match = 0.987)
	f.push_back(128);
	p.push_back(Point(-3,-5));
	f.push_back(128);
	p.push_back(Point(4,2));
	f.push_back(128);
	p.push_back(Point(0,0));//same
	f.push_back(64);
	p.push_back(Point(3,-4));
	f.push_back(64);
	p.push_back(Point(-3,3));
	f.push_back(64);
	p.push_back(Point(-6,6));
	If[4] = Mat::zeros(30,30, CV_8UC1);
	g.display_feature(If[4],f,p,roi2);
	f.clear(); p.clear();
	//5 All points change from 64 to 32 (Cos match = 0.961, Total match = 0.961)
	f.push_back(128);
	p.push_back(Point(-3,-5));
	f.push_back(128);
	p.push_back(Point(4,2));
	f.push_back(32);
	p.push_back(Point(0,0));//same
	f.push_back(32);
	p.push_back(Point(3,-4));
	f.push_back(32);
	p.push_back(Point(-3,3));
	g.display_feature(If[5],f,p,roi);
	f.clear(); p.clear();
	//Depth
	//6
	f.push_back(128);
	p.push_back(Point(-3,-5));
	f.push_back(128);
	p.push_back(Point(4,2));
	f.push_back(64);
	p.push_back(Point(0,0));//same
	f.push_back(64);
	p.push_back(Point(1,0));
	f.push_back(64);
	p.push_back(Point(2,0));
	g.display_feature(If[6],f,p,roi);
	f.clear(); p.clear();
	//7 Middle point goes to zero (Cos match = 0, Total match 0.666)
	f.push_back(128);
	p.push_back(Point(-3,-5));
	f.push_back(128);
	p.push_back(Point(4,2));
	f.push_back(0);
	p.push_back(Point(0,0));//same
	f.push_back(64);
	p.push_back(Point(1,0));
	f.push_back(64);
	p.push_back(Point(2,0));
	f.push_back(128);
	p.push_back(Point(6,0));
	If[7] = Mat::zeros(30,30, CV_8UC1);
	g.display_feature(If[7],f,p,roi2);
	f.clear(); p.clear();
	//8 all pixels are off by 2 (Cos match = 0.854, Total match 0.854)
	f.push_back(128);
	p.push_back(Point(-3,-5));
	f.push_back(128);
	p.push_back(Point(4,2));
	f.push_back(16);
	p.push_back(Point(0,0));//same
	f.push_back(16);
	p.push_back(Point(1,0));
	f.push_back(16);
	p.push_back(Point(2,0));
	g.display_feature(If[8],f,p,roi);
	f.clear(); p.clear();

	//Make dummy modalities
	vector<string> modesGCD,modesDCG;
	modesGCD.push_back("Grad");
	modesGCD.push_back("Color");
	modesGCD.push_back("Depth");

	modesDCG.push_back("Depth");
	modesDCG.push_back("Color");
	modesDCG.push_back("Grad");

	//REORG into modalities
	vector<Mat> T1,T2,T3;
	//Grad,Color,Depth
	T1.push_back(If[0]);
	T1.push_back(If[3]);
	T1.push_back(If[6]);
	//Depth,Color,Grad
	T2.push_back(If[7]);
	T2.push_back(If[4]);
	T2.push_back(If[1]);
	//Grad,Color,Depth
	T3.push_back(If[2]);
	T3.push_back(If[5]);
	T3.push_back(If[8]);

	//TRAIN
	mmod_objects Objs;
	cout << "mmod_objects::learn_a_template #1" << endl;
	string sT1("ST1"), oT("T"), oX("X");
	int num_templ = Objs.learn_a_template(T1,modesGCD, Mask, sT1, oT, 1, (float)1.0);
	cout << "# of templates from first learn = " << num_templ << "\n" << endl;

	cout << "\n\nmmod_objects::learn_a_template #2" << endl;
	Mat Mask2 = Mat::zeros(30,30, CV_8UC1);
	for (int y = 0; y < 30; ++y) {
		if ((y > 9) && (y < 21)) {
			uchar *m = Mask2.ptr<uchar> (y);
			for (int x = 0; x < 30; ++x, ++m) {
				if ((x > 9) && (x < 21)) {
					*m = 1;
				}
			}
		}
	}
	string sT2("ST2");
	num_templ = Objs.learn_a_template(T2,modesDCG, Mask2, sT2, oX, 2, (float)0.5);
	cout << "# of templates from second learn = " << num_templ << "\n" << endl;

	cout << "\nOBJECT MATCHING\n" << endl;
	//Make a mask for one point
	Mat Attend = Mat::zeros(15,11,CV_8UC1);
	Attend.at<uchar>(15/2,11/2) = 1;
	cout << "modesGCD.size() = " << modesGCD.size() << endl;
	int num_obj = Objs.match_all_objects(T3, modesGCD, Attend, 0.8, 0.6, 1,1);
	cout << "num_objs = " << num_obj << endl;
	Objs.cout_matches();
	Mat temp = Mat::zeros(200,200,CV_8UC3);
	Point pp(60,60);
	cout << "Into draw matches ..." << endl;
	Objs.draw_matches(temp,pp);
//	namedWindow("I", 0);
//	imshow("I",temp);
//	waitKey(0);
	////////////////////
	//SERIALIZATION DB/////////////////
	////////////////////
	{
		cout << "Writing models mmod.txt out" << endl;
		std::ofstream ofs("mmod.txt");
		boost::archive::text_oarchive oa(ofs);
		oa << Objs;
	}
    // ----
    mmod_objects Objs2;
    {
    	cout << "Reading models mmod.txt in" << endl;
    	std::ifstream ifs("mmod.txt");
    	boost::archive::text_iarchive ia(ifs);
        // read class state from archive
    	ia >> Objs2;
    }
	///////////////////////////////////
	vector<string> modesGC;
	modesGC.push_back("Grad");
	modesGC.push_back("Color");

	//Construct test images
	Mat Grad = Mat::zeros(150,110, CV_8UC1);
	Mat Color = Mat::zeros(150,110, CV_8UC1);
	Mat Draw = Mat::zeros(150,110, CV_8UC3);
	Mat Maskb = Mat::ones(150,110, CV_8UC1);
	vector<Mat> Test;

	//Make dummy image features
	//Grad
	vector<uchar> fGexact;
	vector<Point> pGexact;
	fGexact.push_back(64);
	pGexact.push_back(Point(0,0));//same
	fGexact.push_back(64);
	pGexact.push_back(Point(-3,-4));
	fGexact.push_back(64);
	pGexact.push_back(Point(3,3));
	g.display_feature_at_Point(Grad,Point(10,10),fGexact,pGexact);
	g.display_feature_at_Point(Grad,Point(50,50),fGexact,pGexact);
	g.display_feature_at_Point(Grad,Point(51,50),fGexact,pGexact);
	g.display_feature_at_Point(Grad,Point(52,50),fGexact,pGexact);
	g.display_feature_at_Point(Grad,Point(53,50),fGexact,pGexact);
	g.display_feature_at_Point(Grad,Point(54,50),fGexact,pGexact);
	g.display_feature_at_Point(Grad,Point(55,50),fGexact,pGexact);
	g.display_feature_at_Point(Color,Point(56,50),fGexact,pGexact);
	g.display_feature_at_Point(Color,Point(57,50),fGexact,pGexact);
	g.display_feature_at_Point(Color,Point(58,50),fGexact,pGexact);
	g.display_feature_at_Point(Color,Point(59,50),fGexact,pGexact);
	g.display_feature_at_Point(Color,Point(60,50),fGexact,pGexact);
	g.display_feature_at_Point(Color,Point(61,50),fGexact,pGexact);
	g.display_feature_at_Point(Color,Point(62,50),fGexact,pGexact);
	g.display_feature_at_Point(Grad,Point(63,50),fGexact,pGexact);
	g.display_feature_at_Point(Grad,Point(64,50),fGexact,pGexact);
	g.display_feature_at_Point(Grad,Point(65,50),fGexact,pGexact);
	g.display_feature_at_Point(Color,Point(66,50),fGexact,pGexact);
	g.display_feature_at_Point(Grad,Point(100,100),fGexact,pGexact);
	//Color
	vector<uchar> fCexact;
	vector<Point> pCexact;
	fCexact.push_back(64);
	pCexact.push_back(Point(0,0));//same
	fCexact.push_back(64);
	pCexact.push_back(Point(3,-4));
	fCexact.push_back(64);
	pCexact.push_back(Point(-3,3));
	g.display_feature_at_Point(Color,Point(10,10),fCexact,pCexact);
	g.display_feature_at_Point(Color,Point(50,50),fCexact,pCexact);
	g.display_feature_at_Point(Color,Point(51,50),fCexact,pCexact);
	g.display_feature_at_Point(Color,Point(52,50),fCexact,pCexact);
	g.display_feature_at_Point(Color,Point(53,50),fCexact,pCexact);
	g.display_feature_at_Point(Color,Point(54,50),fCexact,pCexact);
	g.display_feature_at_Point(Color,Point(55,50),fCexact,pCexact);
	g.display_feature_at_Point(Color,Point(56,50),fCexact,pCexact);
	g.display_feature_at_Point(Color,Point(57,50),fCexact,pCexact);
	g.display_feature_at_Point(Color,Point(58,50),fCexact,pCexact);
	g.display_feature_at_Point(Color,Point(59,50),fCexact,pCexact);
	g.display_feature_at_Point(Color,Point(60,50),fCexact,pCexact);
	g.display_feature_at_Point(Color,Point(61,50),fCexact,pCexact);
	g.display_feature_at_Point(Color,Point(62,50),fCexact,pCexact);
	g.display_feature_at_Point(Color,Point(63,50),fCexact,pCexact);
	g.display_feature_at_Point(Color,Point(64,50),fCexact,pCexact);
	g.display_feature_at_Point(Color,Point(65,50),fCexact,pCexact);
	g.display_feature_at_Point(Color,Point(100,100),fCexact,pCexact);


	//Combine the images
	Test.push_back(Grad);
	Test.push_back(Color);
	int num_objs = Objs2.match_all_objects(Test, modesGC, Maskb, 0.8, 0.6, 1,1);
	cout << "num_objs in large window = " << num_objs << endl;
	Objs2.draw_matches(Draw,Point(0,0));
//	namedWindow("Draw", 0);
//	imshow("Draw",Draw);
//	waitKey(0);

	return num_templ;
}

int main(int argc, char* argv[]) {
	//	mmod_general mytest;
	////	return 0;
	////	fillCosDistTest();
	////	return 0;
	//	Mat msk(Size(200,200),CV_8UC1,Scalar(0,0,0));
	//	Mat feat =	msk.clone();
	//	mmod_general mg;
	//	mmod_features mf;
	//	ellipse(msk,Point(50,60),Size(40,30),30,0,360,Scalar(255),-1);
	//	ellipse(feat,Point(46,55),Size(38,35),0,0,360,Scalar(32),2);
	//	ellipse(feat,Point(56,56),Size(33,22),90,0,360,Scalar(32),2);
	//	ellipse(feat,Point(50,60),Size(40,30),35,0,360,Scalar(64),2);
	//	ellipse(feat,Point(55,65),Size(40,30),127,0,360,Scalar(64),2);
	//	ellipse(feat,Point(50,60),Size(40,30),65,0,360,Scalar(128),2);
	//	ellipse(feat,Point(45,66),Size(40,30),75,0,360,Scalar(128),2);
	////	ellipse(msk,Point(100,120),Size(45,35),150,0,360,Scalar(255),1);
	//	mg.learn_a_template(feat,msk,10,mf);
	//	return 0;
	int num_templ = test_object_rec();
	return 0;

	test_match_a_patch_n_display_feature();
	return 0;
	int board_w = 0;
	int board_h = 0;

	if (argc != 2) {
		cout << "\nERROR: Wrong number of input parameters";
		help();
		return -1;
	}
	//    board_w  = atoi(argv[1]);
	//    board_h  = atoi(argv[2]);

	//    int board_n  = board_w * board_h;
	//   Size board_sz = Size( board_w, board_h );

	FILE* f = fopen(argv[1], "rt");
	if (!f) {
		cout << "\nCouldn't read the file list " << argv[1] << endl;
		help();
		return -1;
	}

	vector < string > filelist;
	for (;;) {
		char buf[1000];
		if (!fgets(buf, (int) sizeof(buf) - 2, f))
			break;
		if (buf[0] == '#' || buf[0] == '\n')
			continue;
		int l = (int) strlen(buf);
		if (buf[l - 1] == '\n')
			buf[l - 1] = '\0';
		filelist.push_back(buf);
	}

	// CAPTURE CORNER VIEWS LOOP UNTIL WE’VE GOT n_boards
	// SUCCESSFUL CAPTURES (ALL CORNERS ON THE BOARD ARE FOUND)
	//
	Size image_size;
	//    Mat gray;

	for (size_t i = 0; i < filelist.size(); i++) {
		Mat image = imread(filelist[i]);
		if (image.empty())
			continue;
		//        cvtColor(image,gray,CV_RGB2GRAY);
		//        threshold(gray,gray,165,255,THRESH_BINARY_INV);
		//        mg.learn_a_template(gray, gray, 10, mf);
		//        imshow( "Preparing for Calibration", image ); //show in color if we did collect the image
		if ((waitKey(0) & 255) == 27)
			return -1;
	} //END COLLECTION WHILE LOOP.
	destroyWindow("Preparing for Calibration");

	//    // SAVE THE INTRINSICS AND DISTORTIONS
	//    cout << " *** DONE!\n\nReprojection error is " << err <<
	//    "\nStoring Intrinsics.xml and Distortions.xml files\n\n";
	//    FileStorage fs("intrinsics.xml", FileStorage::WRITE);
	//
	//    fs << "image_width" << image_size.width << "image_height" << image_size.height <<
	//    "camera_matrix" << intrinsic_matrix << "distortion_coefficients" << distortion_coeffs;
	//    fs.release();
	//
	//    // EXAMPLE OF LOADING THESE MATRICES BACK IN:
	//    fs.open("intrinsics.xml", FileStorage::READ);
	//    cout << "\nimage width: " << (int)fs["image_width"];
	//    cout << "\nimage height: " << (int)fs["image_height"];
	//    Mat intrinsic_matrix_loaded, distortion_coeffs_loaded;
	//    fs["camera_matrix"] >> intrinsic_matrix_loaded;
	//    fs["distortion_coefficients"] >> distortion_coeffs_loaded;
	//    cout << "\nintrinsic matrix:" << intrinsic_matrix_loaded;
	//    cout << "\ndistortion coefficients: " << distortion_coeffs_loaded << endl;

	return 0;
}
