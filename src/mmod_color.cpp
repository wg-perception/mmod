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
#include <utility>
using namespace cv;
using namespace std;

////////////COLOR HLS///////////////////////////////////////////////////////////
/**
 * \brief Compute a color linemod feature based on Hue values near gradients
 * @param Iin  Input BGR image CV_8UC3
 * @param Icolorord Result image CV_8UC1
 * @param Mask  compute on masked region (can be left empty) CV_8UC3 or CV_8UC1 ok
 */
void colorhls::computeColorHLS(const cv::Mat &Iin, cv::Mat &Icolorord, const cv::Mat &Mask)
{
	//CHECK INPUTS
	CALCFEAT_DEBUG_1(cout << "In colorhls::computeColorHLS" << end;);
	if(Itmp.empty() || Iin.rows != Itmp.rows || Iin.cols != Itmp.cols)
	{
		Itmp.create(Iin.size(),CV_8UC3);
	}
	if(Icolorord.empty() || Iin.rows != Icolorord.rows || Iin.cols != Icolorord.cols)
	{
		Icolorord.create(Iin.size(),CV_8UC1);
	}
	Icolorord = Scalar::all(0); //else make sure it's zero
	cv::Mat temp;
	if (!Mask.empty()) //We have a mask
	{
		if (Iin.size() != Mask.size())
		{
			cerr << "ERROR: Mask in computeColorOrder size != Iina" << endl;
			return;
		}
		if (Mask.type() == CV_8UC3)
		{
			//don't write into the Mask, as its supposed to be const.
			cv::cvtColor(Mask, temp, CV_RGB2GRAY);
		}
		else
			temp = Mask;
	}
	//GET HUE
	cvtColor(Iin, Itmp, CV_BGR2HLS);
	vector<Mat> HLS;
	split(Itmp, HLS);
	double minVal = 0,maxVal = 0;
	CALCFEAT_DEBUG_3(
		cout << "HLS size: " << HLS.size() << endl;
		minMaxLoc(HLS[0], &minVal, &maxVal);
		cout << "HLS0 min = " << minVal << ", HLS max = " << maxVal << endl;
		minMaxLoc(HLS[1], &minVal, &maxVal);
		cout << "HLS1 min = " << minVal << ", HLS max = " << maxVal << endl;
		minMaxLoc(HLS[2], &minVal, &maxVal);
		cout << "HLS2 min = " << minVal << ", HLS max = " << maxVal << endl;
	);

	//ONLY REGISTER HUE AROUND STRONG GRADIENTS
	Scharr( HLS[1], grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT ); //dx
	convertScaleAbs( grad_x, abs_grad_x );

	Scharr( HLS[1], grad_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT ); //dy
	convertScaleAbs( grad_y, abs_grad_y );
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );	//|dx| + |dy|
	//Set what "strong" gradient means
	Scalar mean,std;
	meanStdDev(grad, mean, std);
	uchar thresh = (uchar)(mean[0] + std[0]/1.25);
	CALCFEAT_DEBUG_3(
		cout << ", mean = " << mean[0] << ", std = " << std[0] << endl;
		cout << "thresh = " << (int)thresh << endl;
		minMaxLoc(grad, &minVal, &maxVal);
		cout << "grad min = " << minVal << ", grad max = " << maxVal << endl;
		);
	//PRODUCE THE COLOR LINE MOD FEATURE IMAGE
	Mat_<uchar>::iterator h = HLS[0].begin<uchar>(),he = HLS[0].end<uchar>();
	Mat_<uchar>::iterator c = Icolorord.begin<uchar>();
	Mat_<uchar>::iterator g = grad.begin<uchar>();
	if(Mask.empty()) //if no mask
	{
		CALCFEAT_DEBUG_3(cout << "No mask" << endl;);
		for(int i = 0; h != he;++h,++c,++g,++i)
		{
			if((*g) > thresh)	//We only compute colors where gradients are large enough
			{
				int rshift = (int)((float)(*h)/22.5); //Break Hue [0,179] into 8 parts
				*c = (uchar)(1<<rshift);              //Convert this to a single bit
				CALCFEAT_DEBUG_3(*h = (uchar)(rshift*18 + 128););
			}
			else
			{
				*c = 0;
				CALCFEAT_DEBUG_3(*h = 0;);
			}
//			if(*c != 0&&*c != 1&&*c != 2&&*c != 4&&*c!=8&&*c!=16&&*c!=32&&*c!=64&&*c!=128)
//				cout << i<<": bad value of c:"<<(int)*c<<endl;
		}
	} else { //If mask
		Mat_<uchar>::const_iterator m = temp.begin<uchar>();
		CALCFEAT_DEBUG_3(cout << "Use mask" << endl;);
		for(int i = 0; h != he;++h,++c,++g,++m,++i)
		{
			if(!(*m)) continue;  //Only compute pixels with corresponding
			if((*g) > thresh)	 //We only compute colors where gradients are large enough
			{
				int rshift = (int)((float)(*h)/22.5);//Break Hue [0,179] into 8 parts
				*c = (uchar)(1<<rshift);             //Convert this to a single bit
				CALCFEAT_DEBUG_3(*h = (uchar)(rshift*18 + 128););
			}
			else
			{
				*c = 0;
				CALCFEAT_DEBUG_3(*h = 0;);
			}
//			if(*c != 0&&*c != 1&&*c != 2&&*c != 4&&*c!=8&&*c!=16&&*c!=32&&*c!=64&&*c!=128)
//				cout << i<<": bad value of c:"<<(int)*c<<endl;
		}
	}
//	  cout << "colorHLS::computeColorHLS"<<endl;
		c = Icolorord.begin<uchar>();
		for(int i = 0;c != Icolorord.end<uchar>(); ++c,++i)
		{
			if(*c != 0&&*c != 1&&*c != 2&&*c != 4&&*c!=8&&*c!=16&&*c!=32&&*c!=64&&*c!=128)
				cout << i<<": bad value of c:"<<(int)*c<<endl;
		}

	CALCFEAT_DEBUG_3(
		namedWindow("H",0);
		namedWindow("L",0);
		namedWindow("S",0);
		imshow("H",HLS[0]);
		imshow("L",HLS[1]);
		imshow("S",HLS[2]);
		waitKey();
		destroyWindow("H");
		destroyWindow("L");
		destroyWindow("S");
	);
	CALCFEAT_DEBUG_2(cout << "Exit colorhls::computeColorHLS" << end;);
}




////////////////////////GRADIENT FEATURES//////////////////////////////////////////////
/**
 * \brief Compute gradient linemod features from the maximum color plane gradient. Ignores weak gradients
 * @param Iin			Input BGR, CV_8UC3 image
 * @param Icolorord		Output CV_8UC1 image
 * @param Mask			compute on masked region (can be left empty) CV_8UC3 or CV_8UC1 ok
 */
void gradients::computeGradients(const cv::Mat &Iin, cv::Mat &Icolorord, const cv::Mat Mask)
{
	//CHECK INPUTS
	CALCFEAT_DEBUG_1(cout << "In colorhls::computeColorHLS" << end;);
	if(Itmp.empty() || Iin.rows != Itmp.rows || Iin.cols != Itmp.cols)
	{
		Itmp.create(Iin.size(),CV_8UC1);
	}
	if(Icolorord.empty() || Iin.rows != Icolorord.rows || Iin.cols != Icolorord.cols)
	{
		Icolorord.create(Iin.size(),CV_8UC1);
	}
	Icolorord = Scalar::all(0); //else make sure it's zero
	cv::Mat temp;
	if (!Mask.empty()) //We have a mask
	{
		if (Iin.size() != Mask.size())
		{
			cerr << "ERROR: Mask in computeColorOrder size != Iina" << endl;
			return;
		}
		if (Mask.type() == CV_8UC3)
		{
			//don't write into the Mask, as its supposed to be const.
			cv::cvtColor(Mask, temp, CV_RGB2GRAY);
		}
		else
			temp = Mask;
	}
	//FIND THE MAX GRADIENT RESPONSE ACROSS COLORS
//	cvtColor(Iin, Itmp, CV_RGB2GRAY);
	vector<Mat> RGB;
	split(Iin, RGB);

	Scharr( RGB[0], grad_x, CV_32F, 1, 0, 1, 0, BORDER_DEFAULT ); //dx
	Scharr( RGB[0], grad_y, CV_32F, 0, 1, 1, 0, BORDER_DEFAULT ); //dy
	cartToPolar(grad_x, grad_y, mag0, phase0, true); //True => in degrees not radians
	Scharr( RGB[1], grad_x, CV_32F, 1, 0, 1, 0, BORDER_DEFAULT ); //dx
	Scharr( RGB[1], grad_y, CV_32F, 0, 1, 1, 0, BORDER_DEFAULT ); //dy
	cartToPolar(grad_x, grad_y, mag1, phase1, true); //True => in degrees not radians
	Scharr( RGB[2], grad_x, CV_32F, 1, 0, 1, 0, BORDER_DEFAULT ); //dx
	Scharr( RGB[2], grad_y, CV_32F, 0, 1, 1, 0, BORDER_DEFAULT ); //dy
	cartToPolar(grad_x, grad_y, mag2, phase2, true); //True => in degrees not radians

	//COMPUTE RESONABLE THRESHOLDS
	Scalar mean0,std0,mean1,std1,mean2,std2;
	meanStdDev(mag0, mean0, std0);
	meanStdDev(mag1, mean1, std1);
	meanStdDev(mag2, mean2, std2);
#define stdmul 0.4
	float thresh0 = (float)(mean0[0] + std0[0]*stdmul);///1.25);
	float thresh1 = (float)(mean1[0] + std1[0]*stdmul);///1.25);
	float thresh2 = (float)(mean2[0] + std2[0]*stdmul);///1.25);
	double minVal0,maxVal0,minVal1,maxVal1,minVal2,maxVal2;
	CALCFEAT_DEBUG_3(
		cout <<"     means(B,G,R) ("<<mean0[0]<<", "<<mean1[0]<<", "<<mean2[0]<<")"<<endl;
		cout <<"     std(B,G,R)   ("<<std0[0]<<", "<<std1[0]<<", "<<std2[0]<<")"<<endl;
		minMaxLoc(mag0, &minVal0, &maxVal0);
		minMaxLoc(mag1, &minVal1, &maxVal1);
		minMaxLoc(mag2, &minVal2, &maxVal2);
		cout <<"        minVals(B,G,R) ("<<minVal0<<", "<<minVal1<<", "<<minVal2<<")"<<endl;
		cout <<"        maxVals(B,G,R) ("<<maxVal0<<", "<<maxVal1<<", "<<maxVal2<<")"<<endl;
		cout <<"       thresh(B,G,R) ("<<thresh0<<", "<<thresh1<<", "<<thresh2<<")"<<endl;
	);

	//CREATE BINARIZED OUTPUT IMAGE
	MatIterator_<float> mit0 = mag0.begin<float>(), mit_end = mag0.end<float>();
	MatIterator_<float> pit0 = phase0.begin<float>();
	MatIterator_<float> mit1 = mag1.begin<float>();
	MatIterator_<float> pit1 = phase1.begin<float>();
	MatIterator_<float> mit2 = mag2.begin<float>();
	MatIterator_<float> pit2 = phase2.begin<float>();
	MatIterator_<uchar> bit = Icolorord.begin<uchar>();
	float angle;
	if(Mask.empty()) //if no mask
	{
		for(; mit0 != mit_end; ++mit0, ++pit0,++mit1, ++pit1, ++mit2, ++pit2, ++bit)
		{
			if(*mit0 > *mit1)
			{
				if(*mit0 > *mit2) //mit0 is max
				{
					if(*mit0 < thresh0) continue; //Ignore small gradients
					angle = *pit0;
				}
				else // mit2 is max
				{
					if(*mit2 < thresh2) continue;  //Ignore small gradients
					angle = *pit2;
				}
			}
			else if(*mit1 > *mit2)//mit1 is max
			{
				if(*mit1 > *mit2) //mit1 is max
				{
					if(*mit1 < thresh1) continue; //Ignore small gradients
					angle = *pit1;
				}
				else //mit2 is max
				{
					if(*mit2 < thresh2) continue;  //Ignore small gradients
					angle = *pit2;
				}
			}
			if(angle >= 180.0) angle -= 180.0; //We ignore polarity of the angle
			*bit = 1 << (int)(angle*0.044444444); //This is the floor of angle/(180.0/8) to put the angle into one of 8 bits. Set that bit
		}
	}
	else //There is a mask
	{
		Mat_<uchar>::const_iterator m = temp.begin<uchar>();
		for(; mit0 != mit_end; ++mit0, ++pit0,++mit1, ++pit1, ++mit2, ++pit2, ++bit, ++m)
		{
			if(!(*m)) continue;  //Only compute pixels with corresponding mask pixel set
			if(*mit0 > *mit1)
			{
				if(*mit0 > *mit2) //mit0 is max
				{
					if(*mit0 < thresh0) continue; //Ignore small gradients
					angle = *pit0;
				}
				else // mit2 is max
				{
					if(*mit2 < thresh2) continue;  //Ignore small gradients
					angle = *pit2;
				}
			}
			else if(*mit1 > *mit2)//mit1 is max
			{
				if(*mit1 > *mit2) //mit1 is max
				{
					if(*mit1 < thresh1) continue; //Ignore small gradients
					angle = *pit1;
				}
				else //mit2 is max
				{
					if(*mit2 < thresh2) continue;  //Ignore small gradients
					angle = *pit2;
				}
			}
			if(angle >= 180.0) angle -= 180.0; //We ignore polarity of the angle
			*bit = 1 << (int)(angle*0.044444444); //This is the floor of angle/(180.0/8) to put the angle into one of 8 bits. Set that bit
		}
	}
}





#if 0
//////////////////////////////////////////////////////////////////////////////////////////////
// DEFUNCT EFFORTS
//////////////////////////////////////////////////////////////////////////////////////////////

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
void
mmodcolor::computeColorOrder(const cv::Mat &Iin, cv::Mat &Icolorord, const cv::Mat Mask)
{
	cout << "compute color order" << endl;
  if (Iin.size() != Icolorord.size() || Icolorord.type() != CV_8UC1) //Make sure Icolorord is the right size
  {
    Icolorord.create(Iin.size(), CV_8UC1);
  }
  Icolorord = Scalar::all(0); //Zero it
  cv::Mat temp;
  if (!Mask.empty()) //We have a mask
  {
    if (Iin.size() != Mask.size())
    {
      cerr << "ERROR: Mask in computeColorOrder size != Iina" << endl;
      return;
    }
    if (Mask.type() == CV_8UC3)
    {
      //don't write into the Mask, as its supposed to be const.
      cv::cvtColor(Mask, temp, CV_RGB2GRAY);
      //				Mask = temp;
    }
    else
      temp = Mask;
    if (Mask.type() != CV_8UC1)
    {
      cerr << "ERROR: Mask is not of type CV_8UC1 in computeColorOrder" << endl;
    }
  }
  uchar white = (uchar) (255 - black_white_thresh);
  int cnt[256];
  for(int i = 0; i<256; ++i)
	  cnt[i] = 0;
  int foo = 0;
  if (temp.empty()) //No mask
  {
    for (int y = 0; y < Iin.rows; ++y)
    {
      const uchar *b = Iin.ptr<uchar> (y);
      const uchar *g = b + 1;
      const uchar *r = b + 2;
      uchar *o = Icolorord.ptr<uchar> (y);
      for (int x = 0; x < Iin.cols; x++, b += 3, g += 3, r += 3, o++)
      {
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
        if ((*b < black_white_thresh) && (*g < black_white_thresh) && (*r < black_white_thresh))
        {
          *o = 64; //Black
          cnt[64] += 1;
          continue;
        }
        if ((*b > white) && (*g > white) && (*r > white))
        {
        	cnt[128] += 1;
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
            cnt[1] += 1;
            continue;
          }
          if (*r > *b) //R > B (>= G)
          {
        	  cnt[16] += 1;
            *o = 16;
            continue;
          }
        }
        if (br > -equal_thresh) //B>=R
        {
          if (gr < equal_thresh) //(B>=) R>=G (-gr > -equal_thresh
          {
        	  cnt[2] += 1;
            *o = 2;
            continue;
          }
          if (*g > *b) //G > B (>=R)
          {
        	  cnt[4] += 1;
            *o = 4;
            continue;
          }
        }
        if (gr > -equal_thresh) //G>=R
        {
          if (*r > *b) //(G>=) R > B
          {
        	  cnt[8] += 1;
            *o = 8;
            continue;
          }
        }
        if ((*r > *g) && (*g > *b))
        {
        	cnt[32] += 1;
          *o = 32;
          continue;
        }
 //       if(!foo%1000)
        	cout << (int)*r << ", " << (int)*g << ", " << (int)*b << endl;
        	foo++;
        cnt[10] += 1;
 //       cerr << "ERROR: No color value should be zero" << endl;
  //      return;
      }
    }
  }//end if no mask
  else //Use mask
  {
    for (int y = 0; y < Iin.rows; y++)
    {
      const uchar *b = Iin.ptr<uchar> (y);
      const uchar *g = b + 1;
      const uchar *r = b + 2;
      const uchar *m = temp.ptr<uchar> (y);
      uchar *o = Icolorord.ptr<uchar> (y);
      for (int x = 0; x < Iin.cols; x++, b += 3, g += 3, r += 3, o++)
      {
        if (!(*m))
        {
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
        if ((*b < black_white_thresh) && (*g < black_white_thresh) && (*r < black_white_thresh))
        {
          *o = 64; //Black
          cout << "black" << endl;
          continue;
        }
        if ((*b > white) && (*g > white) && (*r > white))
        {
        	cout << "White" << endl;
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
        	  cnt[1] +=1;
            *o = 1; //B>=G>=R
            continue;
          }
          if (*r > *b) //R > B (>= G)
          {
        	  cnt[16] += 1;
            *o = 16;
            continue;
          }
        }
        if (br > -equal_thresh) //B>=R
        {
          if (gr < equal_thresh) //(B>=) R>=G (-gr > -equal_thresh
          {
        	  cnt[2] += 1;
            *o = 2;
            continue;
          }
          if (*g > *b) //G > B (>=R)
          {
        	  cnt[4] += 1;
            *o = 4;
            continue;
          }
        }
        if (gr > -equal_thresh) //G>=R
        {
          if (*r > *b) //(G>=) R > B
          {
        	  cnt[8] += 1;
            *o = 8;
            continue;
          }
        }
        if ((*r > *g) && (*g > *b))
        {
        	cnt[32] += 1;
          *o = 32;
          continue;
        }
        cnt[12] += 1;
  //      cerr << "ERROR: No color value should be zero outside of the mask" << endl;
 //       return;
      }
    }
  }//#end mask
  for(int i = 0; i<256; ++i)
	  cout << i << ": " << cnt[i] << ", ";
  cout << endl;
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
void
colorwta::computeColorWTA(const cv::Mat &Iin, cv::Mat &Icolorord, const cv::Mat Mask)
{
	CALCFEAT_DEBUG_1(cout << "In colorwta::computeColorWTA, Iin.rows,cols = "<<Iin.rows<<", "<<Iin.cols<<endl;);
	if(Iin.type() != CV_8UC3)
	{
		cerr << "ERROR: in colorwta::computeColorWTA, Iin does not have type CV_8UC3, it has type: " << Iin.type()<<endl;
		return;
	}
	if (Iin.size() != Icolorord.size() || Icolorord.type() != CV_8UC1) //Make sure Icolorord is the right size
	{
		Icolorord.create(Iin.size(), CV_8UC1);
	}
	Icolorord = Scalar::all(0); //Zero it
	cv::Mat temp;
	if (!Mask.empty()) //We have a mask
	{
		if (Iin.size() != Mask.size())
		{
			cerr << "ERROR: Mask in computeColorOrder size != Iina" << endl;
			return;
		}
		if (Mask.type() == CV_8UC3)
		{
			//don't write into the Mask, as its supposed to be const.
			cv::cvtColor(Mask, temp, CV_RGB2GRAY);
			//				Mask = temp;
		}
		else
			temp = Mask;
		if (Mask.type() != CV_8UC1)
		{
			cerr << "ERROR: Mask is not of type CV_8UC1 in computeColorOrder" << endl;
		}
	}
	int cnt[256];
	CALCFEAT_DEBUG_3(
			for(int i = 0; i<256; ++i)
				cnt[i] = 0;
	);
	Idst = Iin;
	GaussianBlur(Iin, Idst2, Size(5, 5), 2);
//	GaussianBlur(Idst,Idst, Size(5,5), 2);
//	GaussianBlur(Idst, Idst2, Size(5, 5), 2);
//	GaussianBlur(Idst2,Idst2, Size(5,5),2);
//	GaussianBlur(Idst, Idst, Size(5, 5), 2);
	CALCFEAT_DEBUG_3(cout << "After GaussianBlur, Idst.rows,cols = " << Idst.rows<<", "<<Idst.cols<<endl;);
	uchar vals[8];
	int b1,b2,g1,g2,r1,r2,dg,black,white,yellow,rl,gr,ru,gd;
	if (temp.empty()) //No mask
	{

		CALCFEAT_DEBUG_3(cout << "No mask"<<endl;);
//		for (int y = 3; y < Idst.rows - 4; y++)
		for (int y = 1; y < Idst.rows - 1; y++)
		{
			uchar *o = (Icolorord.ptr<uchar> (y)) + 1;// + 3;
			Vec3b *ga1 = Idst.ptr<Vec3b> (y) + 1;
			Vec3b *ga2 = Idst2.ptr<Vec3b> (y) + 1;
//			for (int x = 3; x < Idst.cols - 4; x++, o++)
			for (int x = 1; x < Idst.cols - 1; x++, o++, ++ga1, ++ga2)
			{
				b1 = (int)((*ga1)[0]);
				b2 = (int)((*ga2)[0]);
				g1 = (int)((*ga1)[1]);
				g2 = (int)((*ga2)[1]);
				r1 = (int)((*ga1)[2]);
				r2 = (int)((*ga2)[2]);
				int index = 0;
				//r-g
				int max = r1 - g2;
				//g-r
				dg = g1 - r2;
				if(max < dg) {max = dg; index = 1;}
				//y-b
				yellow = (g1+r1)>>1;
				dg = yellow - b2;
				if(max < dg) {max = dg; index = 2;}
				//b-y
				yellow = (g2+r2)>>1;
				dg = b1 - yellow;
				if(max < dg) {max = dg; index = 3;}
				//white-black
				white = (r1+g1+b1)/3;
				black = ((r2+g2+b2)/3);
				dg = white - black;
				if(max < dg) {max = dg; index = 4;}
				//black-white
				if(max < -dg) {max = -dg; index = 5;}
				dg = (int)((*(ga1 - 1))[2]) - (int)((*(ga1 + 1))[1]);
				//dx_rg
				if(dg < 0) dg = -dg;
				if(max < dg) {max = dg; index = 6;}
				//dy_rg
				dg = (int)(Idst.at<Vec3b> (y-1,x)[2]) - (int)(Idst.at<Vec3b> (y+1,x)[1]);
				if(dg < 0) dg = -dg;
				if(max < dg) {max = dg; index = 7;}


//				vals[0] = Idst.at<Vec3b> (y - 2, x + 3)[2];
//				vals[1] = Idst.at<Vec3b>     (y, x - 3)[0];
//				vals[2] = Idst.at<Vec3b> (y + 1, x + 1)[0];
//				vals[3] = Idst.at<Vec3b> (y - 3, x - 3)[1];
//				vals[4] = Idst.at<Vec3b>     (y - 2, x)[2];
//				vals[5] = Idst.at<Vec3b> (y - 1, x + 1)[0];
//				vals[6] = Idst.at<Vec3b> (y + 2, x + 2)[1];
//				vals[7] = Idst.at<Vec3b> (y - 1, x - 1)[1];

//				uchar max = 0;
//				int index = 0;
//				for (int i = 0; i < 8; ++i)
//				{
//					if (vals[i] > max)
//					{
//						max = vals[i];
//						index = i;
//					}
//				}
				CALCFEAT_DEBUG_3(cnt[1<<index] += 1;);
				CALCFEAT_DEBUG_4(
						if(!(y%40)&&(!(x%40)))
						{
							cout << "vals: ";
							for(int i = 0; i<8; ++i)
							{
								cout << i <<":"<<(int)vals[i]<<" ";
							}
							cout << "\n max is "<<(int)max<<" at index "<<index<<endl;
							cout << "which has byte rep "<< (int)(1<<index)<< endl;
						}
				);
				*o = (uchar)1 << index;

			}
		}
	}
	else //Use mask
	{
		cout << "Mask" << endl;
		CALCFEAT_DEBUG_3(cout << "With mask"<<endl;);
		for (int y = 1; y < Idst.rows - 1; y++)
		{

			const uchar *m = temp.ptr<uchar> (y) + 1;
			uchar *o = Icolorord.ptr<uchar> (y) + 1;
			Vec3b *ga1 = Idst.ptr<Vec3b> (y) + 1;
			Vec3b *ga2 = Idst2.ptr<Vec3b> (y) + 1;

			for (int x = 1; x < Idst.cols - 1; x++, o++,++m,++ga1,++ga2)
			{
				if (!(*m))
					continue;
				b1 = (int)((*ga1)[0]);
				b2 = (int)((*ga2)[0]);
				g1 = (int)((*ga1)[1]);
				g2 = (int)((*ga2)[1]);
				r1 = (int)((*ga1)[2]);
				r2 = (int)((*ga2)[2]);
				int index = 0;
				//r-g
				int max = r1 - g2;
				//g-r
				dg = g1 - r2;
				if(max < dg) {max = dg; index = 1;}
				//y-b
				yellow = (g1+r1)>>1;
				dg = yellow - b2;
				if(max < dg) {max = dg; index = 2;}
				//b-y
				yellow = (g2+r2)>>1;
				dg = b1 - yellow;
				if(max < dg) {max = dg; index = 3;}
				//white-black
				white = (r1+g1+b1)/3;
				black = ((r2+g2+b2)/3);
				dg = white - black;
				if(max < dg) {max = dg; index = 4;}
				//black-white
				if(max < -dg) {max = -dg; index = 5;}
				dg = (int)((*(ga1 - 1))[2]) - (int)((*(ga1 + 1))[1]);
				//dx_rg
				if(dg < 0) dg = -dg;
				if(max < dg) {max = dg; index = 6;}
				//dy_rg
				dg = (int)(Idst.at<Vec3b> (y-1,x)[2]) - (int)(Idst.at<Vec3b> (y+1,x)[1]);
				if(dg < 0) dg = -dg;
				if(max < dg) {max = dg; index = 7;}

				//				vals[0] = Idst.at<Vec3b> (y - 2, x + 3)[2];
//				vals[1] = Idst.at<Vec3b> (y, x - 3)[0];
//				vals[2] = Idst.at<Vec3b> (y + 1, x + 1)[0];
//				vals[3] = Idst.at<Vec3b> (y - 3, x - 3)[1];
//				vals[4] = Idst.at<Vec3b> (y - 2, x)[2];
//				vals[5] = Idst.at<Vec3b> (y - 1, x + 1)[0];
//				vals[6] = Idst.at<Vec3b> (y + 2, x + 2)[1];
//				vals[7] = Idst.at<Vec3b> (y - 1, x - 1)[1];
//				uchar max = 0;
//				int index = 0;
//				for (int i = 0; i < 8; ++i)
//				{
//					if (vals[i] > max)
//					{
//						max = vals[i];
//						index = i;
//					}
//				}
				CALCFEAT_DEBUG_3(cnt[1<<index] += 1;);

				CALCFEAT_DEBUG_4(
						if(!(y%40)&&(!(x%40)))
						{
							cout << "vals: ";
							for(int i = 0; i<8; ++i)
							{
								cout << i <<":"<<(int)vals[i]<<" ";
							}
							cout << "\n max is "<<(int)max<<" at index "<<index<<endl;
							cout << "which has byte rep "<< (int)((uchar)1<<index)<< endl;
						}
				);
				*o = (uchar)1 << index;
			}
		}
	}
	CALCFEAT_DEBUG_3(
			for(int i = 0; i< 8; ++i)
			{
				int foo = 1<<i;
				cout <<i<<"("<<foo<<"): "<< cnt[1<<i] << ", " << endl;
			});
	CALCFEAT_DEBUG_2(cout << "Exit colorwta::computeColorWTA()"<<endl;
	imshow("ColorMap",Icolorord);
	);
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
void
depthwta::computeDepthWTA(const cv::Mat &Iin, cv::Mat &Icolorord, const cv::Mat Mask)
{
  CALCFEAT_DEBUG_1(
      cout << "In depthwta::computeDepthWTA Iin,rows,cols = "<<Iin.rows<<", "<<Iin.cols<<endl;);

  if (Iin.size() != Icolorord.size() || Icolorord.type() != CV_8UC1) //Make sure Icolorord is the right size
  {
    Icolorord.create(Iin.size(), CV_8UC1);
  }
  Icolorord = Scalar::all(0); //Zero it
  cv::Mat temp;
  if (!Mask.empty()) //We have a mask
  {
    if (Iin.size() != Mask.size())
    {
      throw std::logic_error("ERROR: Mask in computeDepthWTA size != Iina");
    }
    if (Mask.type() == CV_8UC3)
    {
      //don't write into the Mask, as its supposed to be const.
      cv::cvtColor(Mask, temp, CV_RGB2GRAY);
    }
    else
      temp = Mask;
  }
  cv::Mat tIin;
  if (Iin.type() == CV_32FC1) // this means we're in meters, need to quantize to mm
  {
    Iin.convertTo(tIin, CV_16U, 1000.0);// meters -> mm
  }else
    tIin = Iin;

  if (tIin.type() != CV_16UC1)
  {
    throw std::logic_error("You must convert the depth map to CV_16UC1 for computeDepthWTA");
  }
  GaussianBlur(tIin, Idst, Size(5, 5), 2);
  ushort vals[8];
  if (temp.empty()) //No mask
  {
    for (int y = 3; y < Idst.rows - 4; y++)
    {
      uchar *o = Icolorord.ptr<uchar> (y) + 3;
      for (int x = 3; x < Idst.cols - 4; x++, o++)
      {
        vals[0] = Idst.at<ushort> (y - 2, x + 3);
        vals[1] = Idst.at<ushort> (y, x - 3);
        vals[2] = Idst.at<ushort> (y + 1, x + 1);
        vals[3] = Idst.at<ushort> (y - 3, x - 3);
        vals[4] = Idst.at<ushort> (y - 2, x);
        vals[5] = Idst.at<ushort> (y - 1, x + 1);
        vals[6] = Idst.at<ushort> (y + 2, x + 2);
        vals[7] = Idst.at<ushort> (y - 1, x - 1);
        ushort max = 0;
        int index = 0;
        for (int i = 0; i < 8; ++i)
        {
          if (vals[i] > max)
          {
            max = vals[i];
            index = i;
          }
        }
        *o = 1 << index;
      }
    }
  }
  else //Use mask
  {
    for (int y = 3; y < Idst.rows - 4; y++)
    {

      const uchar *m = temp.ptr<uchar> (y) + 3;
      uchar *o = Icolorord.ptr<uchar> (y) + 3;
      for (int x = 3; x < Idst.cols - 4; x++, o++,++m)
      {
        if (!(*m))
          continue;
        vals[0] = Idst.at<ushort> (y - 2, x + 3);
        vals[1] = Idst.at<ushort> (y, x - 3);
        vals[2] = Idst.at<ushort> (y + 1, x + 1);
        vals[3] = Idst.at<ushort> (y - 3, x - 3);
        vals[4] = Idst.at<ushort> (y - 2, x);
        vals[5] = Idst.at<ushort> (y - 1, x + 1);
        vals[6] = Idst.at<ushort> (y + 2, x + 2);
        vals[7] = Idst.at<ushort> (y - 1, x - 1);
        ushort max = 0;
        int index = 0;
        for (int i = 0; i < 8; ++i)
        {
          if (vals[i] > max)
          {
            max = vals[i];
            index = i;
          }
        }
        *o = 1 << index;
      }
    }
  }
  CALCFEAT_DEBUG_2(
      cout << "Exit..."<<endl;
      imshow("ColorMap",Icolorord);
  );
}
#endif
