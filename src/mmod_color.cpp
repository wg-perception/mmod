/*
 * mmod_color.cpp
 *
 * Compute line mod color features
 *
 *  Created on: Sep 13, 2011
 *      Author: Gary Bradski
 */
#include "mmod_color.h"
#include <iostream>
#include <stdexcept>

using namespace cv;
using namespace std;
//////////////////////////////////////////////////////////////////////////////////////////////
/**
 * mmodcolor  Produces color coded images, each pixel converted to one of 8 bits:
 * bit Color Order  number
 * 0   B >= G >= R  0
 * 1   B >= R >= G  2
 * 2   G >  B >= R  4
 * 3   G >= R >  B  8
 * 4   R >  B >= G 16
 * 5   R >  G >  B 32
 * 6   All colors < black_white_thresh       => black  64
 * 7   All colors > 255 - black_white_thresh => white 128
 *
 * Colors are considered "equal if within equal_thresh of one another
 */

	/**
	 * void computeColorOrder(const cv::Mat &Iina, cv::Mat &Icolorord, const cv::Mat &Mask)
	 *
	 * Code a color input image into a single channel (one byte) coded color image using an optional mask
	 *
	 * @param Iina        Input BGR image of uchars
	 * @param Icolorord   Return coded image here
	 * @param Mask        (optional) skip masked pixels (0's) if Mask is supplied.
	 */
	void mmodcolor::computeColorOrder(const cv::Mat &Iin, cv::Mat &Icolorord, const cv::Mat Mask) {
		if (Iin.size() != Icolorord.size() || Icolorord.type() != CV_8UC1) //Make sure Icolorord is the right size
		{
			Icolorord.create(Iin.size(), CV_8UC1);
		}
		Icolorord = Scalar::all(0); //Zero it
		cv::Mat temp;
		if (!Mask.empty()) //We have a mask
		{
			if (Iin.size() != Mask.size()) {
				cerr << "ERROR: Mask in computeColorOrder size != Iina" << endl;
				return;
			}
			if (Mask.type() == CV_8UC3) {
				//don't write into the Mask, as its supposed to be const.
				cv::cvtColor(Mask, temp, CV_RGB2GRAY);
//				Mask = temp;
			}
			else temp = Mask;
			if (Mask.type() != CV_8UC1) {
				cerr
						<< "ERROR: Mask is not of type CV_8UC1 in computeColorOrder"
						<< endl;
			}
		}
		uchar white = (uchar)(255 - black_white_thresh);
		if (temp.empty()) //No mask
		{
			for (int y = 0; y < Iin.rows; ++y) {
				const uchar *b = Iin.ptr<uchar> (y);
				const uchar *g = b + 1;
				const uchar *r = b + 2;
				uchar *o = Icolorord.ptr<uchar> (y);
				for (int x = 0; x < Iin.cols; x++, b += 3, g += 3, r += 3, o++) {
					/* bit Color Order
					 * 0   B >= G >= R  1
					 * 1   B >= R >= G  2
					 * 2   G >  B >= R  4
					 * 3   G >= R >  B  8
					 * 4   R >  B >= G 16
					 * 5   R >  G >  B 32
					 * 6   All colors < black_white_thresh       => black 64
					 * 7   All colors > 255 - black_white_thresh => white 128 */
					//black
					if ((*b < black_white_thresh) && (*g < black_white_thresh)
							&& (*r < black_white_thresh)) {
						*o = 64; //Black
						continue;
					}
					if ((*b > white) && (*g > white) && (*r > white)) {
						*o = 128; //White
						continue;
					}
					//B >= G
					int bg = *b - *g;
					int gr = *g - *r;
					int br = *b - *r;
					if (bg > -equal_thresh) //B >= G
					{
						if (gr > -equal_thresh) //G >= R
						{
							*o = 1; //B>=G>=R
							continue;
						}
						if (*r > *b) //R > B (>= G)
						{
							*o = 16;
							continue;
						}
					}
					if (br > -equal_thresh) //B>=R
					{
						if (gr < equal_thresh) //(B>=) R>=G (-gr > -equal_thresh
						{
							*o = 2;
							continue;
						}
						if (*g > *b) //G > B (>=R)
						{
							*o = 4;
							continue;
						}
					}
					if (gr > -equal_thresh) //G>=R
					{
						if (*r > *b) //(G>=) R > B
						{
							*o = 8;
							continue;
						}
					}
					if ((*r > *g) && (*g > *b)) {
						*o = 32;
						continue;
					}
					cerr << "ERROR: No color value should be zero" << endl;
					return;
				}
			}
		}//end if no mask
		else //Use mask
		{
			for (int y = 0; y < Iin.rows; y++) {
				const uchar *b = Iin.ptr<uchar> (y);
				const uchar *g = b + 1;
				const uchar *r = b + 2;
				const uchar *m = temp.ptr<uchar> (y);
				uchar *o = Icolorord.ptr<uchar> (y);
				for (int x = 0; x < Iin.cols; x++, b += 3, g += 3, r += 3, o++) {
					if (!(*m)) {
						*o = 0; //This combination indicates that the bit is invalid/masked
						continue;
					}
					/* bit Color Order
					 * 0   B >= G >= R  1
					 * 1   B >= R >= G  2
					 * 2   G >  B >= R  4
					 * 3   G >= R >  B  8
					 * 4   R >  B >= G 16
					 * 5   R >  G >  B 32
					 * 6   All colors < black_white_thresh       => black 64
					 * 7   All colors > 255 - black_white_thresh => white 128 */
					//black
					if ((*b < black_white_thresh) && (*g < black_white_thresh)
							&& (*r < black_white_thresh)) {
						*o = 64; //Black
						continue;
					}
					if ((*b > white) && (*g > white) && (*r > white)) {
						*o = 128; //White
						continue;
					}
					//B >= G
					int bg = *b - *g;
					int gr = *g - *r;
					int br = *b - *r;
					if (bg > -equal_thresh) //B >= G
					{
						if (gr > -equal_thresh) //G >= R
						{
							*o = 1; //B>=G>=R
							continue;
						}
						if (*r > *b) //R > B (>= G)
						{
							*o = 16;
							continue;
						}
					}
					if (br > -equal_thresh) //B>=R
					{
						if (gr < equal_thresh) //(B>=) R>=G (-gr > -equal_thresh
						{
							*o = 2;
							continue;
						}
						if (*g > *b) //G > B (>=R)
						{
							*o = 4;
							continue;
						}
					}
					if (gr > -equal_thresh) //G>=R
					{
						if (*r > *b) //(G>=) R > B
						{
							*o = 8;
							continue;
						}
					}
					if ((*r > *g) && (*g > *b)) {
						*o = 32;
						continue;
					}
					cerr
							<< "ERROR: No color value should be zero outside of the mask"
							<< endl;
					return;
				}
			}
		}//#end mask
	}

//////////////////WTA////////////////////////////
	/**
	 * \brief Compute a winner take all inspired color feature.
	 *
	 * Basically it does Gaussian blur and then chooses the max index from 8 points in a 7x7 patch around each pixel.
	 *
	 * @param Iin 			Input image (color)
	 * @param Icolorord		Return CV_8UC1 image where each uchar codes for the max index in patch around each pixel
	 * @param Mask			CV_8UC1 or CV_8UC3 region to collect from. Can be empty (no mask).
	 */
	void colorwta::computeColorWTA(const cv::Mat &Iin, cv::Mat &Icolorord, const cv::Mat Mask) {
		if (Iin.size() != Icolorord.size() || Icolorord.type() != CV_8UC1) //Make sure Icolorord is the right size
		{
			Icolorord.create(Iin.size(), CV_8UC1);
		}
		Icolorord = Scalar::all(0); //Zero it
		cv::Mat temp;
		if (!Mask.empty()) //We have a mask
		{
			if (Iin.size() != Mask.size()) {
				cerr << "ERROR: Mask in computeColorOrder size != Iina" << endl;
				return;
			}
			if (Mask.type() == CV_8UC3) {
				//don't write into the Mask, as its supposed to be const.
				cv::cvtColor(Mask, temp, CV_RGB2GRAY);
//				Mask = temp;
			}
			else temp = Mask;
			if (Mask.type() != CV_8UC1) {
				cerr << "ERROR: Mask is not of type CV_8UC1 in computeColorOrder" << endl;
			}
		}
		GaussianBlur(Iin, Idst, Size(5,5), 2);
		uchar vals[8];
		if (temp.empty()) //No mask
		{
			for (int y = 3; y < Idst.rows-4; y++) {
				uchar *o = Icolorord.ptr<uchar> (y);
				for (int x = 3; x < Idst.cols-4; x++, o++){
					vals[0] = Idst.at<Vec3b>(y-2,x+3)[2];
					vals[1] = Idst.at<Vec3b>(y,x-3)[0];
					vals[2] = Idst.at<Vec3b>(y+1,x+1)[0];
					vals[3] = Idst.at<Vec3b>(y-3,x-3)[1];
					vals[4] = Idst.at<Vec3b>(y-2,x)[2];
					vals[5] = Idst.at<Vec3b>(y-1,x+1)[0];
					vals[6] = Idst.at<Vec3b>(y+2,x+2)[1];
					vals[7] = Idst.at<Vec3b>(y-1,x-1)[1];
					uchar max = 0;
					int index = 0;
					for(int i = 0; i<8; ++i)
					{
						if(vals[i] > max){max = vals[i]; index = i;}
					}
					*o = 1<<index;
				}
			}
		}
		else //Use mask
		{
			for (int y = 3; y < Idst.rows-4; y++) {

				const uchar *m = temp.ptr<uchar> (y);
				uchar *o = Icolorord.ptr<uchar> (y);
				for (int x = 3; x < Idst.cols-4; x++, o++){
					if(!(*m)) continue;
					vals[0] = Idst.at<Vec3b>(y-2,x+3)[2];
					vals[1] = Idst.at<Vec3b>(y,x-3)[0];
					vals[2] = Idst.at<Vec3b>(y+1,x+1)[0];
					vals[3] = Idst.at<Vec3b>(y-3,x-3)[1];
					vals[4] = Idst.at<Vec3b>(y-2,x)[2];
					vals[5] = Idst.at<Vec3b>(y-1,x+1)[0];
					vals[6] = Idst.at<Vec3b>(y+2,x+2)[1];
					vals[7] = Idst.at<Vec3b>(y-1,x-1)[1];
					uchar max = 0;
					int index = 0;
					for(int i = 0; i<8; ++i)
					{
						if(vals[i] > max){max = vals[i]; index = i;}
					}
					*o = 1<<index;
				}
			}
		}
	}

//////////////wta depth///////////////////////////////
	typedef Vec<ushort, 3> Vec3us;

	/**
	 * \brief Compute a winner take all inspired depth feature.
	 *
	 * Basically it does Gaussian blur and then chooses the max index from 8 points in a 7x7 patch around each pixel.
	 *
	 * @param Iin 			Input CV_16UC1 depth image
	 * @param Icolorord		Return CV_8UC1 image where each uchar codes for the max index in the patch around that pixel
	 * @param Mask			CV_8UC1 or CV_8UC3 region to collect from. Can be empty (no mask).
	 */
	void depthwta::computeDepthWTA(const cv::Mat &Iin, cv::Mat &Icolorord, const cv::Mat Mask) {
		if (Iin.size() != Icolorord.size() || Icolorord.type() != CV_8UC1) //Make sure Icolorord is the right size
		{
			Icolorord.create(Iin.size(), CV_8UC1);
		}
		Icolorord = Scalar::all(0); //Zero it
		cv::Mat temp;
		if (!Mask.empty()) //We have a mask
		{
			if (Iin.size() != Mask.size()) {
				throw std::logic_error("ERROR: Mask in computeDepthWTA size != Iina");
			}
			if (Mask.type() == CV_8UC3) {
				//don't write into the Mask, as its supposed to be const.
				cv::cvtColor(Mask, temp, CV_RGB2GRAY);
			}
			else
			  temp = Mask;
		}
		if(Iin.type() != CV_16UC1)
		{
		  throw std::logic_error("You must convert the depth map to CV_16UC1 for computeDepthWTA");
		}
		GaussianBlur(Iin, Idst, Size(5,5), 2);
		ushort vals[8];
		if (temp.empty()) //No mask
		{
			for (int y = 3; y < Idst.rows-4; y++) {
				uchar *o = Icolorord.ptr<uchar> (y);
				for (int x = 3; x < Idst.cols-4; x++, o++){
					vals[0] = Idst.at<ushort>(y-2,x+3);
					vals[1] = Idst.at<ushort>(y,x-3);
					vals[2] = Idst.at<ushort>(y+1,x+1);
					vals[3] = Idst.at<ushort>(y-3,x-3);
					vals[4] = Idst.at<ushort>(y-2,x);
					vals[5] = Idst.at<ushort>(y-1,x+1);
					vals[6] = Idst.at<ushort>(y+2,x+2);
					vals[7] = Idst.at<ushort>(y-1,x-1);
					ushort max = 0;
					int index = 0;
					for(int i = 0; i<8; ++i)
					{
						if(vals[i] > max){max = vals[i]; index = i;}
					}
					*o = 1<<index;
				}
			}
		}
		else //Use mask
		{
			for (int y = 3; y < Idst.rows-4; y++) {

				const uchar *m = temp.ptr<uchar> (y);
				uchar *o = Icolorord.ptr<uchar> (y);
				for (int x = 3; x < Idst.cols-4; x++, o++){
					if(!(*m)) continue;
					vals[0] = Idst.at<ushort>(y-2,x+3);
					vals[1] = Idst.at<ushort>(y,x-3);
					vals[2] = Idst.at<ushort>(y+1,x+1);
					vals[3] = Idst.at<ushort>(y-3,x-3);
					vals[4] = Idst.at<ushort>(y-2,x);
					vals[5] = Idst.at<ushort>(y-1,x+1);
					vals[6] = Idst.at<ushort>(y+2,x+2);
					vals[7] = Idst.at<ushort>(y-1,x-1);
					ushort max = 0;
					int index = 0;
					for(int i = 0; i<8; ++i)
					{
						if(vals[i] > max){max = vals[i]; index = i;}
					}
					*o = 1<<index;
				}
			}
		}
	}
