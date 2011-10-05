/*
 * mmod_color.h
 *
 *  Created on: Sep 13, 2011
 *      Author: Gary Bradski
 */

#ifndef MMOD_COLOR_H_
#define MMOD_COLOR_H_
#include <opencv2/opencv.hpp>

#define CALCFEAT_SHOW() {std::cout << __FILE__ << " : "  << __LINE__ << std::endl;}
  //VERBOSE
  // 1 Routine list, 2 values out, 3 internal values outside of loops, 4 intenral values in loops
  #define CALCFEAT_VERBOSE 0

  #if CALCFEAT_VERBOSE >= 1
  #define CALCFEAT_DEBUG_1(X) do{CALCFEAT_SHOW() X }while(false)
  #else
  #define CALCFEAT_DEBUG_1(X) do{}while(false)
  #endif

  #if CALCFEAT_VERBOSE >= 2
  #define CALCFEAT_DEBUG_2(X) do{CALCFEAT_SHOW() X}while(false)
  #else
  #define CALCFEAT_DEBUG_2(X) do{}while(false)
  #endif
  #if CALCFEAT_VERBOSE >= 3
  #define CALCFEAT_DEBUG_3(X) do{X}while(false)
  #else
  #define CALCFEAT_DEBUG_3(X) do{}while(false)
  #endif
  #if CALCFEAT_VERBOSE >= 4
  #define CALCFEAT_DEBUG_4(X) do{CALCFEAT_SHOW() X}while(false)
  #else
  #define CALCFEAT_DEBUG_4(X) do{}while(false)
  #endif


//////////////////////////////////////////////////////////////////////////////////////////////
/**  This is a "line mod type way of classing color
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
class mmodcolor {
public:
	unsigned equal_thresh; //If colors are within this distance, they are considered equal
	unsigned black_white_thresh; //If all colors are within their min/max value and this bound, call them black/white
	mmodcolor() :
		equal_thresh(12), black_white_thresh(25) {}	;
	mmodcolor(unsigned et, unsigned bwt) {equal_thresh = et;
		black_white_thresh = bwt;};

	/**
	 * void computeColorOrder(const cv::Mat &Iina, cv::Mat &Icolorord, const cv::Mat &Mask)
	 *
	 * Code a color input image into a single channel (one byte) coded color image using an optional mask
	 *
	 * @param Iina        Input BGR image of uchars
	 * @param Icolorord   Return coded image here
	 * @param Mask        (optional) skip masked pixels (0's) if Mask is supplied.
	 */
	void computeColorOrder(const cv::Mat &Iin, cv::Mat &Icolorord, cv::Mat Mask);
};

////////////COLOR HLS///////////////////////////////////////////////////////////
class colorhls {
	cv::Mat Itmp;
	cv::Mat grad_x, grad_y, grad;
	cv::Mat abs_grad_x, abs_grad_y;
public:
	/**
	 * \brief Compute a color linemod feature based on Hue values near gradients
	 * @param Iin  Input BGR image CV_8UC3
	 * @param Icolorord Result image CV_8UC1
	 * @param Mask  compute on masked region (can be left empty) CV_8UC3 or CV_8UC1 ok
	 */
	void computeColorHLS(const cv::Mat &Iin, cv::Mat &Icolorord, const cv::Mat Mask);
};

////////////COLOR WTA/////////////////////////////////////////////////////////////
/**
 * This second method is inspired by the "winner take all" method from the paper
 * "The Power of Comparative Reasoning
 */
class colorwta {
	cv::Mat Idst, Idst2;
public:
	/**
	 * \brief Compute a winner take all inspired color feature.
	 *
	 * Basically it does Gaussian blur and then chooses the max index from 8 points in a 7x7 patch around each pixel.
	 *
	 * @param Iin 			Input image (color)
	 * @param Icolorord		Return CV_8UC1 image where each uchar codes for the max index in patch around each pixel
	 * @param Mask			CV_8UC1 or CV_8UC3 region to collect from. Can be empty (no mask).
	 */
	void computeColorWTA(const cv::Mat &Iin, cv::Mat &Icolorord, const cv::Mat Mask);
};


////////////COLOR WTA/////////////////////////////////////////////////////////////
/**
 * This class produces uchar depth features where each feature (at each pixel) comes from a Gaussian blur
 * followed by the maximum index of 8 randomly chosen points around the corresponding pixel.
 */
class depthwta {
	cv::Mat Idst;
public:
	/**
	 * \brief Compute a winner take all inspired depth feature.
	 *
	 * Basically it does Gaussian blur and then chooses the max index from 8 points in a 7x7 patch around each pixel.
	 *
	 * @param Iin 			Input CV_16UC1 depth image
	 * @param Icolorord		Return CV_8UC1 image where each uchar codes for the max index in the patch around that pixel
	 * @param Mask			CV_8UC1 or CV_8UC3 region to collect from. Can be empty (no mask).
	 */
	void computeDepthWTA(const cv::Mat &Iin, cv::Mat &Icolorord, const cv::Mat Mask);
};

#endif /* MMOD_COLOR_H_ */
