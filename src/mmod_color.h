/*
 * mmod_color.h
 *
 *  Created on: Sep 13, 2011
 *      Author: vadmin
 */

#ifndef MMOD_COLOR_H_
#define MMOD_COLOR_H_

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

/**
 * This second method is inspired by the "winner take all" method from the paper
 * "The Power of Comparative Reasoning
 */
class colorwta {
public:
	int K;		//Window length
};

#endif /* MMOD_COLOR_H_ */
