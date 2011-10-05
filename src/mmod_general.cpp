/*
 * mmod_general.cpp
 *
 *  Created on: Sep 8, 2011
 *      Author: Gary Bradski
 */
#include "mmod_general.h"
using namespace cv;
using namespace std;
//////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * \brief mmod_general constructor. Fills Cos distances in matchLUT.
	 *
	 * @return
	 */
mmod_general::mmod_general()
	{
		fillCosDist();
	}

	/**
	 * \brief Take a mmod_features derived uchar feature vector and copy it out to an image structure. This is for matching features to features and for debug.
	 *
	 * Input image I must be single channel uchar and of adequate size to display the whole feature (no bounds checking is done).
	 * The feature is drawn relative to the center pixel so the bounding box should be offset from there. For example, for a 50x60 (width,height) image and
	 * a feature with bounding box of bbwidth 10, bbheight 20, we'd set x = -10/2; and y= -20/2. This easily fits within the image bounds.
	 * To make sure it is of adequate size, just make sure mmod_features::max_bounds.width < I.cols and mmod_features::max_bounds.height < I.rows.
	 *
	 * Example call: display_feature(patch, ftemp.features[index], ftemp.offsets[index], ftemp.bbox[index]); //will draw this feature into "patch"
	 *
	 * @param I		Input image in which to display the feature (must be CV_8UC1 and large enough to contain the feature offsets
	 * @param f		A uchar feature vector
	 * @param o		The offsets from center pixel of each uchar in the feature vector above
	 * @param bbox  The bounding box of this feature
	 */
	void mmod_general::display_feature(Mat &I, vector<uchar> &f, vector<Point> &o, Rect &bbox)
	{
	  GENL_DEBUG_1(cout<<"mmod_general::display_feature. Size features:"<<f.size()<<" size offsets:"<<o.size()<<endl;);
		Rect R = bbox;
		R.x += I.cols/2; R.y += I.rows/2; //Just put the roi against the upper left edge
		GENL_DEBUG_3(cout << "R(" << R.x << ", "<< R.y << ", "<< R.width << ", "<< R.height << ")" << endl;
		cout << "I(" << I.cols << ", " << I.rows <<")" << endl;);
		Mat Iroi = I(R); //Work within our region of interest
		GENL_DEBUG_3(cout << "Iroi(" << Iroi.cols << ", " << Iroi.rows << ")" << endl;);
		Iroi = Scalar::all(0); //Clear out that region
		vector<uchar>::iterator _fit;			//feature vals iterator
		vector<Point>::iterator _oit; 			//offset values iterator
		int xc = Iroi.cols/2, yc = Iroi.rows/2;
		GENL_DEBUG_3(cout << "(xc,yc) = " << xc << ", " << yc << ")" << endl;);
		//PUT OUR CURRENT FEATURE INTO THE IMAGE.
		int i = 0;
		for(_fit = f.begin(), _oit = o.begin(); _fit != f.end(); ++_fit, ++_oit, ++i)
		{
		  GENL_DEBUG_4(cout << "i=" << i << ", x = " << xc + (*_oit).x << ", y = " << yc + (*_oit).y << endl;);
			Iroi.at<uchar>(yc + (*_oit).y, xc + (*_oit).x) = *_fit;
		}
	}

	/**
	 * \brief This takes in binarized 8U_C1 image and colorizes it for visualization.
	 *
	 * Colors are Bright red 128, dull red 64, bright yellow 32, dull yellow 16, bright purple 8,
	 * dull purple 4, blue 2, dull blue 1, black 0
	 *
	 * @param I		input binarized image [128,64,32,16,8,4,2,1,0] CV_8UC1
	 * @param iB	output colorized image CV_8UC3
	 */
	void mmod_general::visualize_binary_image(const cv::Mat &I, cv::Mat &iB)
	{
		if(iB.empty()||iB.rows != I.rows || iB.cols != I.cols)
		{
			iB.create(I.size(),CV_8UC3);
		}
		iB = Scalar::all(0);
		uchar c;
//		int cnt[256];
//		for(int i = 0; i<256; ++i)
//			cnt[i] = 0;
		for(int y = 1; y < I.rows; y++) {
			uchar *bptr = iB.ptr<uchar>(y);
			const uchar *cptr = I.ptr<uchar>(y);
			for(int x = 0; x < I.cols; x++, bptr += 3, ++cptr)
			{
				c = *cptr;
				uchar blues = (uchar)((float)(c&7)*18.0 + 128.0);
				if((c<<5) == 0) blues = 0;
//				uchar blues = c << 5;
				uchar greens = (uchar)(float((c>>3)&7)*18.0 + 128.0);
				if((c & 0x38) == 0) greens = 0;
//				uchar greens = (c & 0x38)<<2;
//				uchar reds = c & 0xC0;
				uchar reds = (uchar)((float)((c>>5)&3)*18.0 + 128.0);
				if((c & 0xC0) == 0) reds = 0;
				*bptr = blues;
				*(bptr + 1) = greens;
				*(bptr + 2) = reds;

			}
		}
	}


	/**
	 * \brief Draw a feature (vector<uchar> with offsets (vector<Point>) into an image at a point. Does bounds checking
	 *
	 * This function is used for debug (testing matching)
	 *
	 * @param I		Input image, should be CV_8UC1
	 * @param p		Point to draw feature relative to
	 * @param f		Vector of feature values
	 * @param o		Vector of offsets
	 */
	void mmod_general::display_feature_at_Point(Mat &I, Point p, vector<uchar> &f, vector<Point> &o)
	{
//		I = Scalar::all(0); //Clear out that region
		vector<uchar>::iterator _fit;			//feature vals iterator
		vector<Point>::iterator _oit; 			//offset values iterator
		int xc = p.x, yc = p.y;
		int rows = I.rows, cols = I.cols; //for bounds
		//PUT OUR CURRENT FEATURE INTO THE IMAGE.
		int i = 0;
		for(_fit = f.begin(), _oit = o.begin(); _fit != f.end(); ++_fit, ++_oit, ++i)
		{
			int Y = yc + (*_oit).y;
			if((Y < 0)||(Y >= rows)) continue;
			int X = xc + (*_oit).x;
			if((X < 0)||(X >= cols)) continue;
			I.at<uchar>(Y, X) = *_fit;
		}
	}

	/**
	 * \brief Brute force match linemod templates at (centered on) a particular point in an image
	 *
	 * This is mainly called via mmod_objects::match_all_objects
	 *
	 * Do a brute force match of all the templates in a given mmod_features at a given pixel (Point) in an image.  It does bounds checking for you.
	 *
	 * @param I				Input image or patch
	 * @param p				Point(x,y) at which to match
	 * @param f				trained mmod_features reference to match against
	 * @param match_index   which feature had the maximal match score
	 * @return				score of maximal match. If f is empty, return 0 (nothing matches)
	 */
	float mmod_general::match_a_patch_bruteforce(const Mat &I, const Point &p, mmod_features &f, int &match_index)
	{
	  GENL_DEBUG_1(cout<<"mmod_general::match_a_patch_bruteforde"<<endl;);
		match_index = -1;
		if(f.features.empty())//Handle edge conditions
		{
		  GENL_DEBUG_2(cout << "g:match a patch features empty" << endl;);
			return(0.0);
		}

		vector <Rect>::iterator rit;			//Rectangle iterator
		vector <vector<uchar> >::iterator fit;	//feature set iterator (each set of features)
		vector<uchar>::iterator _fit;			//feature vals iterator (within the the current *rit bounding box)
		vector<vector<Point> >::iterator oit;	//offset set iterator (each set of offsets)
		vector<Point>::iterator _oit; 			//offset values iterator (offsets to each feature within the bbox set)

		int k;
		float match, maxmatch = 0;
		int norm;
		int rows = I.rows, cols = I.cols;
		Rect imgRect(0,0,cols,rows);

		//FOR FEATURES
		for(rit = f.bbox.begin(), fit = f.features.begin(), oit = f.offsets.begin(), k=0; rit != f.bbox.end(); ++fit, ++oit, ++rit, ++k)
		{
			Rect Rpatch(p.x + (*rit).x,p.y + (*rit).y,(*rit).width,(*rit).height);
			match = 0;
			norm = 0;
			GENL_DEBUG_4(
				cout << "Len of features: " << (*fit).size() << endl;
			);
			Rect Ri = imgRect & Rpatch; //Intersection between patch and image
			int Risize = Ri.width * Ri.height;
			int Rpsize = Rpatch.width * Rpatch.height;
			if(Risize == Rpsize) //Intersection between patch and image is the same size at patch
			{
//				cout << "Good:" << Ri.width << "vs" << Rpatch.width << ", "<< Ri.height<< "vs" << Rpatch.height << endl;
				GENL_DEBUG_4(cout << "NO BOUNDS CHECK NEEDED" << endl;);
				for(_fit = (*fit).begin(), _oit = (*oit).begin(); _fit != (*fit).end(); ++_fit, ++_oit)
				{
				    GENL_DEBUG_4(
				    		cout << "*_fit:" << (int)(*_fit) << " at( " << p.y + (*_oit).y << ", " << p.x + (*_oit).x << "), I= " << (int)(I.at<uchar>(p.y + (*_oit).y, p.x + (*_oit).x)) << endl;
				    );
					match += matchLUT[lut[*_fit]][I.at<uchar>(p.y + (*_oit).y, p.x + (*_oit).x)]; //matchLUT[lut[model_uchar]][test_uchar]
					GENL_DEBUG_4(
							cout << "matchLUT = " << matchLUT[lut[*_fit]][I.at<uchar>(p.y + (*_oit).y, p.x + (*_oit).x)] << endl;
					);
					++norm;
				}
			}
			else //bounds checking needed
			{
				if(Risize < (int)(Rpsize*0.7)) //Don't try to match too small of areas at the edge
				{
					norm = 1; match = 0.01;
				}
				else
				{
					//				cout << "Bad:" << Ri.width << "vs" << Rpatch.width << ", "<< Ri.height<< "vs" << Rpatch.height << endl;

					GENL_DEBUG_4(cout << "BOUNDS CHECKING NEEDED" << endl;);
					for(_fit = (*fit).begin(), _oit = (*oit).begin(); _fit != (*fit).end(); ++_fit, ++_oit)
					{
						int xx = p.x + (*_oit).x;  //bounds check the indices
						if((xx < 0)||(xx >= cols)) continue;
						int yy = p.y + (*_oit).y;
						if((yy < 0)||(yy >= rows)) continue;

						match += matchLUT[lut[*_fit]][I.at<uchar>(yy,xx)]; //matchLUT[lut[model_uchar]][test_uchar]
						++norm;
					}
				}
			}

			GENL_DEBUG_4(cout << "norm in g:match_a_patch = " << norm << endl;);
			if(0 == norm) norm = 1;
			match /= (float)norm;
			if(match > maxmatch)
			{
				maxmatch = match;
				match_index = k;
			}
		}
		GENL_DEBUG_2(cout << "Max match = "<<maxmatch<<" norm="<<norm<<endl;);
		return maxmatch;
	}




	/**
	 * \brief Brute force match a linemod filter template at (centered on) a particular point in an image
	 *
	 * This is the matching function for mmod_filters to return a score of a given object/modality and given view at a point
	 *
	 * @param I				Input image or patch
	 * @param R				Rectangle at which to match
	 * @param f				trained mmod_features reference to match against
	 * @param index   		index of which view to use in mmod_features
	 * @return				score of match. If f is empty, return 0 (nothing matches)
	 */
	float mmod_general::match_one_feature(const Mat &I, const Rect &R, mmod_features &f, int index)
	{
		GENL_DEBUG_1(cout<<"mmod_general::match_one_feature"<<endl;);

		if(f.features.empty())//Handle edge conditions
		{
			GENL_DEBUG_2(cout << "f.features is empty" << endl;);
			return(0.0);
		}
		Point p(R.x + R.width/2, R.y + R.height/2);
//		f.features[index];
//		f.offsets[index];

		int k;
		float match = 0;
		int norm = 0;
		int rows = I.rows, cols = I.cols;
		Rect imgRect(0,0,cols,rows);
		vector<uchar>::iterator _fit;			//feature vals iterator (within the the current *rit bounding box)
		vector<Point>::iterator _oit; 			//offset values iterator (offsets to each feature within the bbox set)

		Rect Ri = imgRect & R; //Intersection between patch and image
		int Risize = Ri.width * Ri.height;
		int Rpsize = R.width * R.height;
		if(Risize == Rpsize) //Intersection between patch and image is the same size at patch
		{
			GENL_DEBUG_4(cout << "NO BOUNDS CHECK NEEDED" << endl;);
			for(_fit = f.features[index].begin(), _oit = f.offsets[index].begin(); _fit != f.features[index].end(); ++_fit, ++_oit)
			{
				match += matchLUT[lut[*_fit]][I.at<uchar>(p.y + (*_oit).y, p.x + (*_oit).x)]; //matchLUT[lut[model_uchar]][test_uchar]
				++norm;
			}
		}
		else //bounds checking needed
		{
			if(Risize < (int)(Rpsize*0.7)) //Don't try to match too small of areas at the edge
			{
				norm = 1; match = 0.01;
			}
			else
			{
				//				cout << "Bad:" << Ri.width << "vs" << Rpatch.width << ", "<< Ri.height<< "vs" << Rpatch.height << endl;

				GENL_DEBUG_4(cout << "BOUNDS CHECKING NEEDED" << endl;);
				for(_fit = f.features[index].begin(), _oit = f.offsets[index].begin(); _fit != f.features[index].end(); ++_fit, ++_oit)
				{
					int xx = p.x + (*_oit).x;  //bounds check the indices
					if((xx < 0)||(xx >= cols)) continue;
					int yy = p.y + (*_oit).y;
					if((yy < 0)||(yy >= rows)) continue;

					match += matchLUT[lut[*_fit]][I.at<uchar>(yy,xx)]; //matchLUT[lut[model_uchar]][test_uchar]
					++norm;
				}
			}//End else not too little rectangle left in scene
		}//End bounds checking needed

		if(0 == norm) norm = 1;
		match /= (float)norm;

		GENL_DEBUG_2(cout << "Score = "<<match<<" norm="<<norm<<endl;);
		return match;
	}











	/**
	 * Given an 8UC1 image where each pixel is a byte with at most 1 bit on, Either:
	 * 0 OR into each pixel the spanXspan values surrounding that pixel, Or
	 * 1 calculate the bit in the majority within each span x span window and output that "cleaned up" image
	 *
	 *  void SumAroundEachPixel8UC1(Mat &co, Mat &out, int span)
	 *  co  -- input 8UC1 image where each pixel is a byte with at most 1 bit on
	 *  out -- output "cleaned up" image (can be the same as co and is faster that way)
	 *  span -- the size of the spanXspan window in which to calulate the majority
	 *  Or0_Max1 -- If 0, compute the span x span OR, else compute the Majority bit type in a span x span window.
	 */
	void mmod_general::SumAroundEachPixel8UC1(Mat &co, Mat &out, int span, int Or0_Max1)
	{
		//Allocate or reallocate accumulation arrays
		if(8 != (int)acc.size())
		{
			acc.resize(8);
			acc2.resize(8);
		}
		if((acc[0].empty())||(co.size() != acc[0].size()))
		{
			for(int i = 0; i<8; ++i)
			{
				acc[i].create(co.size(),CV_32SC1);
				acc2[i].create(co.size(),CV_32SC1);
			}
		}
		if(out.empty()||(out.size() != co.size())||(out.type()!=co.type()))
		{
			out.create(co.size(),co.type());
		}

		//SUM ACROSS ROWS
		int span05 = span/2;
		vector<int *> a(8); //accumulators
		vector <int *>::iterator it;

		for (int y = 0; y < co.rows; y++)
		{

			//Input
			uchar *o = co.ptr<uchar> (y);
			//Sum span
			uchar *s = o;
			uchar *e = o;
			//Accum pointer set and set first position to zero
			int i;
			for(i = 0, it = a.begin(); it != a.end(); ++it, ++i)
			{
				*it = acc[i].ptr<int> (y); //Set to point at their row start
				*(*it) = 0;
			}

			//[1]All counts out to half the window span apply to the first element of the accumulator. Window start is not moved
			for(int x = 0; x<=span05; ++x, ++e)
			{
				if(*e != 0)
					*(a[lut[*e]]) += 1; //Accumulate a hit for this bit
			}
			//Move the accumulator pointers across a column
			for(it=a.begin(); it != a.end(); ++it)
				*it += 1;
			//[2]All counts from half the window out to the window span do not move the starting pointer but do move the accumulator pointers
			for(int x = span05+1; x < span; ++x, ++e)
			{
				//Copy the previous accumulation to the current position
				for(it=a.begin(); it != a.end(); ++it)
					*(*it) = *((*it)-1);
				//Sum in the new value
				if(*e != 0)
				{
					*(a[lut[*e]]) += 1; //accumulate
				}
				//Increment the accumulator pointers
				for(it=a.begin(); it != a.end(); ++it)
					*it += 1;
			}

			//[3]Now we move all pointers until we come within span05 of the end
			for (int x = span; x < co.cols; ++x, ++s, ++e)
			{
				//Copy the previous accumulation to the current position
				for(it=a.begin(); it != a.end(); ++it)
					*(*it) = *((*it)-1);
				//Sum in the new value
				if(*e != 0)
					*(a[lut[*e]]) += 1; //accumulate
				if(*s != 0)
					*(a[lut[*s]]) -= 1; //Get rid of trailing edge of accumulation window

				//Increment the accumulator pointers
				for(it=a.begin(); it != a.end(); ++it)
					*it += 1;
			}

			//[4] Finally, the end of the window is off the edge of the image, so we unwind the rest of the window
			for(int x = co.cols-span05; x<co.cols; ++x, ++s)
			{
				//Copy the previous accumulation to the current position
				for(it=a.begin(); it != a.end(); ++it)
					*(*it) = *((*it)-1);
				//Sum in the new value
				if(*s != 0)
				{
					*(a[lut[*s]]) -= 1; //Get rid of trailing edge of accumulation window
				}
				//Increment the accumulator pointers
				for(it=a.begin(); it != a.end(); ++it)
					*it += 1;
			}
		}//End step down y


		//SUM THE ACCUMULATOR ARRAYS ACROSS COLS
		vector<int *> as(8),ae(8);  //Start and end pointers for windows
		vector <int *>::iterator aeit,asit;
		int jj;
		for (int x = 0; x < acc[0].cols; ++x)
		{
			//Sum over a column window of len span, set the pointers to top of column x
			for(jj = 0, aeit = ae.begin(), asit = as.begin(); aeit != ae.end(); ++aeit,++asit,++jj)
			{
				*asit = acc[jj].ptr<int>(0) + x;
				*aeit = *asit;
			}
			//Set pointers into the acc2 array and set first position to zero
			int i;
			for(i = 0, it = a.begin(); it != a.end(); ++it, ++i)
			{
				*it = acc2[i].ptr<int> (0) + x; //Set to point at their column start
				*(*it) = 0;
			}

			//[1]All counts out to half the window span apply to the first element of the accumulator. Left side of window is not moved
			for(int y = 0; y<=span05; ++y)
			{
				//Accumulate step  y as window slides down
				for(it = a.begin(), aeit = ae.begin(); it != a.end(); ++it, ++aeit)
				{
					*(*it) += *(*aeit);
					*aeit += acc[0].step1();
				}
			}
			//Move the accumulator pointers down a row
			for(it=a.begin(); it != a.end(); ++it)
				*it += acc2[0].step1();

			//[2]All counts from half the window out to the window span do not move the starting pointer but do move the accumulator pointers
			for(int y = span05+1; y < span; ++y)
			{
				//Copy the previous accumulation one row up to the current position
				for(it=a.begin(); it != a.end(); ++it)
					*(*it) = *((*it)-acc2[0].step1());
				//Sum in the new values
				for(it = a.begin(), aeit = ae.begin(); it != a.end(); ++it, ++aeit)
				{
					*(*it) += *(*aeit);
				}
				//Increment the accumulator pointers
				for(it=a.begin(); it != a.end(); ++it)
					*it += acc2[0].step1();
				//Increment the end pointers
				for(aeit = ae.begin(); aeit != ae.end(); ++aeit)
					*aeit += acc[0].step1();
			}

			//[3]Now we move all pointers until we come within span05 of the end row
			for (int y = span; y < co.rows; ++y)
			{
				//Copy the previous one row up accumulation to the current position
				for(it=a.begin(); it != a.end(); ++it)
					*(*it) = *((*it)-acc2[0].step1());

				//Sum in the new value on bottom of window, subtract old value on top
				for(it = a.begin(), aeit = ae.begin(), asit = as.begin(); it != a.end(); ++it, ++aeit, ++asit)
				{
					*(*it) += *(*aeit);
					*(*it) -= *(*asit);
				}
				//Increment the accumulator pointers to the next row
				for(it=a.begin(); it != a.end(); ++it)
					*it += acc2[0].step1();
				//Increment the window start and end pointers
				for(aeit = ae.begin(),asit = as.begin(); aeit != ae.end(); ++aeit,++asit)
				{
					*aeit += acc[0].step1();
					*asit += acc[0].step1();
				}
			}

			//[4] Finally, the end of the window is off the edge of the image, so we unwind the rest of the window
			for(int y = co.rows-span05; y<co.rows; ++y)
			{
				//Copy the previous accumulation one row up to the current position
				for(it=a.begin(); it != a.end(); ++it)
					*(*it) = *((*it)-acc2[0].step1());
				//Get rid of the tailing edge of the accumulation window
				for(it = a.begin(), asit = as.begin(); it != a.end(); ++it, ++asit)
					*(*it) -= *(*asit);
				//Increment the accumulator pointers
				for(it=a.begin(); it != a.end(); ++it)
					*it += acc2[0].step1();
				//Increment the window start and end pointers
				for(asit = as.begin(); asit != as.end(); ++asit)
					*asit += acc[0].step1();
			}


		}//End step across x

		//OUTPUT:
		if(Or0_Max1){//FIND MAX and put into output
			int max, maxpos,pos;
			for (int y = 0; y < out.rows; y++)
			{
				//Output
				uchar *o = out.ptr<uchar> (y);
				uchar *in = co.ptr<uchar> (y);
				//Set the accumulation pointers
				int i;
				for(i = 0, it = a.begin(); it != a.end(); ++it, ++i)
				{
					*it = acc2[i].ptr<int> (y); //Set to point at their row start
				}
				//For each col position in this row, replace it by the majority bit value within the spanXspan window
				for (int x = 0; x < co.cols; ++x, ++o, ++in)
				{
					//Find the max orientation
					for(it=a.begin(), max = 0, maxpos = 0, pos = 0; it != a.end(); ++it, ++pos)
					{
						if(*(*it) > max)
						{
							max = *(*it);
							maxpos = pos;
						}
					}
					// output the majority value within the spanXspan window
					if((max > 1)&&(*in))//||(!(*in) && max == 1))
						*o = 1<<maxpos;
					else
						*o = *in; //Retain previous input
					//Increment the accumulator pointers across columns in this row
					for(it=a.begin(); it != a.end(); ++it)
						*it += 1;
				}
			}
		} else {//FIND THE SPAN X SPAN "OR" AT EACH PIXEL AND RECORD IT INTO output
			for (int y = 0; y < out.rows; y++)
			{
				//Output
				uchar *o = out.ptr<uchar> (y);
				uchar *in = co.ptr<uchar> (y);
				//Set the accumulation pointers
				int i;
				for(i = 0, it = a.begin(); it != a.end(); ++it, ++i)
				{
					*it = acc2[i].ptr<int> (y); //Set to point at their row start
				}
				//For each col position in this row, replace it by the majority bit value within the spanXspan window
				int pos;

				for (int x = 0; x < co.cols; ++x, ++o, ++in)
				{
						*o = 0;
						//Find the max orientation
						for(it=a.begin(), pos = 0; it != a.end(); ++it, ++pos)
						{
							if(*(*it) > 0)
							{
								*o |= 1<<pos;
							}
						}
					//Increment the accumulator pointers across columns in this row
					for(it=a.begin(); it != a.end(); ++it)
						*it += 1;
				}
			}
		}
	}//End SumAroundEachPixel8UC1 method


	/**
	 *\brief fillCosDist() -- fill up the match lookup table with COS distance functions
	 *
	 * This just fills up matchLUT[9][256] with single bit set byte to byte Cos match look up
	 * for model (single bit set byte) m, and image uchar byte b, the match would be looked up as
	 * matchLUT[lut[m]][b]
	 */
	void mmod_general::fillCosDist()
	{
//		if(lut[2] != 1)
//		{
	  GENL_DEBUG_1(cout <<"In mmod_general::fillCosDist()"<<endl;);
			for(int k = 0; k<256; ++k) lut[k] = 8;//illegal value for accum arrays, will cause error but shouldn't be hit
			lut[0] = 8;
			lut[1] = 0;
			lut[2] = 1;
			lut[4] = 2;
			lut[8] = 3;
			lut[16] = 4;
			lut[32] = 5;
			lut[64] = 6;
			lut[128] = 7;
//		}
		matchLUT.resize(9);
		uchar a, al, ar; //left and right shift
		float dist[9] = {
			    1.000000000,//(cos(0)+1)/2      0 (dist in bits)
				0.961939766,//(cos(22.5)+1)/2   1
				0.853553391,//(cos(45)+1)/2     2
				0.691341716,//(cos(67.5)+1)/2   3
			    0.500000000,//(cos(90)+1)/2     4
				0.308658284,//(cos(112.5)+1)/2  5
				0.146446609,//(cos(135)+1)/2    6
				0.038060234,//(cos(157.5)+1)/2  7
				0.000000000//(cos(180)+1)/2     8
		};
		int s;
		for(int v = 0; v<8; ++v) //For each bit position 0 through 7
		{
			matchLUT[v].resize(256);
			a = 1<<v;
			for(unsigned int i = 0; i<256; ++i) //For each possible byte value
			{
				for(s = 0; s<8; ++s) //For all possible shift distances
				{
					al = a<<s;
					ar = a>>s;
					if(al & i) break; //Stop when we find a match
					if(ar & i) break;
				}
				matchLUT[v][i] = dist[s]; //This will be accessed as matchLUT[lut[model_offset]][image_uchar]
			}
		}
		matchLUT[8].resize(256);
		for(int i = 0; i<256; ++i)
			matchLUT[8][i] = 0.0;
	}


	/**
	 * \brief Match a uchar to a uchar using the match look up table
	 *
	 * This function is really just an example of how matchLUT is called inner loop.
	 *
	 * @param model_uchar  From the model (a single bit is set only)
	 * @param image_uchar  From the feature image
	 * @return 			Match value
	 */
	float mmod_general::match(uchar &model_uchar, uchar &image_uchar)
	{
		return matchLUT[lut[model_uchar]][image_uchar];
	}


	/**
	 * \brief Non-Maximum Suppression: Suppress overlapping rectangles in favor of the rectangle with the highest score
	 *
	 * @param rv			vector of rectangle to check
	 * @param scores		vector of their match scores (keep the largest score preferentially)
	 * @param object_ID		vector of class names associated with the rectangle(s)
	 * @param frame_number	vector of frame_number number associated with the rectangles
	 * @param feature_indices This is a vector of features for each mode. mode[].vector<int>. Look up is then modes[modality].objs[name].features[match_index]
	 * @param frac_overlap	the fraction of overlap between 2 rectangles that constitutes "overlap"
	 * @return 				Num of rectangles cleaned of overlap left in rv.
	 */
	int  mmod_general::nonMaxRectSuppress(vector<Rect> &rv, vector<float> &scores, vector<string> &object_ID,
			vector<int> &frame_number, std::vector<std::vector<int> > &feature_indices, float frac_overlap)
	{
		int len = (int)rv.size();
		if( len != (int)scores.size()) { cerr << "ERROR nonMaxRectSuppress has missmatched lengths" << endl; return -1;}
		Rect ri;
		for(int i = 0; i<len - 1; ++i)
		{
			for(int j = i+1; j<len; ++j)
			{
			    float total_area = rv[i].height*rv[i].width + rv[j].height*rv[j].width + 0.000001; //prevent divide by zero
			    ri = rv[i] & rv[j]; //Rectangle intersection
			    float measured_frac_overlap = (2.0*ri.height*ri.width/(total_area));  //Return the fraction of intersection
				if(measured_frac_overlap > frac_overlap)
				{
					if(scores[i] >= scores[j])
					{
						rv.erase(rv.begin()+j);
						scores.erase(scores.begin()+j);
						object_ID.erase(object_ID.begin()+j);
						frame_number.erase(frame_number.begin()+j);
						feature_indices.erase(feature_indices.begin()+j);
						len -= 1;
						j -= 1;
						if(i >= len)
							break;
					}
					else
					{
						rv.erase(rv.begin()+i);
						scores.erase(scores.begin()+i);
						object_ID.erase(object_ID.begin()+i);
	                    frame_number.erase(frame_number.begin()+i);
	                    feature_indices.erase(feature_indices.begin()+i);
						len -= 1;
						i -= 1;
						break;
					}
				}
			}
		}
		return (int)rv.size();
	}



	/**
	 * \brief Given a binarized feature image and a mask of where to collect features, learn a template there (no matter if other templates match it well).
	 *
	 * This is mainly called via mmod_objects::learn_a_template.
	 *
	 * We often use this by passing in a temporary features class with no entries.  We learn the feature and decide
	 * to include it in our real mmod_features only if no existing template matches it well
	 * (found by using mmod_general::display_feature and then mmod_general::match_a_patch_bruteforce or by using
	 * scalable matching).
	 *
	 * @param Ifeatures   Input image 8U_C1 of binarized features
	 * @param Mask		  Mask 8U_C1 of where the object is
	 * @param framenum	  frame number of this view
	 * @param features	  this will hold our learned template
	 * @param clean		  If true, do a 3x3 max filter to the features. Default is false
	 * @return index of template learned in features variable.
	 */
	int mmod_general::learn_a_template(Mat &Ifeatures,  Mat &Mask, int framenum, mmod_features &features, bool clean)
	{
	  GENL_DEBUG_1(cout << "In mmod_general::learn_a_template. framenum:"<<framenum<<" At start: features.features.size() = " << features.features.size() << endl;);
		if (clean) SumAroundEachPixel8UC1(Ifeatures, Ifeatures, 3, 1); //This sets the middle pixel (if it's not 0) to the max of its 3x3 surround
	    //FIND MAX CONTOUR
	    vector<vector<Point> > contours;
	    vector<Vec4i> hierarchy;
	    findContours( Mask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
	    int numcontours = contours.size();
	    int maxc = 0, maxpos;
	    for(int i = 0; i<numcontours; ++i)
	    {
	    	int cs = contours[i].size();
	    	if(cs > maxc){ maxc = cs; maxpos = i;}
	    }
	    //FIND CENTER
	    Rect R = boundingRect(contours[maxpos]);
	    GENL_DEBUG_3(cout << "Found contour bbox = (" << R.x << ", " << R.y << ", " << R.width << ", " << R.height << ")" <<endl;);
	    int xc = R.width/2, yc = R.height/2;
	    features.frame_number.push_back(framenum);

	    Mat Ifroi = Ifeatures(R), Imroi = Mask(R);  //Just process around the needed sites
	    R.x = -xc; R.y = -yc; //This sets up the rectangle to be offset around the pixel that is being examined
	    //Update the enclosing bbox of features.bbox
	    if(R.width > features.max_bounds.width) { features.max_bounds.width = R.width; features.max_bounds.x = R.x;}
	    if(R.height > features.max_bounds.height) { features.max_bounds.height = R.height; features.max_bounds.y = R.y;}

	    int featN = features.features.size() + 1;
	    vector<uchar> fv;
	    vector<Point> ov;
	    vector<int> UL,UR,LL,LR;
	    int index = 0;
	    GENL_DEBUG_3(cout << "Total possible pushbacks = " << R.width*R.height << endl;);
	    int mcnt = 0,fcnt = 0;
		for (int y = 0; y < Imroi.rows; y++)
		{
			//Output
			uchar *f = Ifroi.ptr<uchar> (y);
			uchar *m = Imroi.ptr<uchar> (y);
			for (int x = 0; x < Imroi.cols; ++x, ++f, ++m)
			{
				GENL_DEBUG_3(
					if(*m) ++mcnt;
					if(*f) ++fcnt;
				);

				if(*m && *f)
				{
					int ox = x-xc, oy = y-yc; //Feature offsets are relative to the center
					ov.push_back(Point(ox,oy));
					//Find quadrent of this point
					if(oy<0) //U
					{
						if(ox<0) //UL
						{
							UL.push_back(index);
						}
						else //UR
						{
							UR.push_back(index);
						}
					}
					else //L
					{
						if(ox<0) //LL
						{
							LL.push_back(index);
						}
						else //LR
						{
							LR.push_back(index);
						}
					}
					fv.push_back(*f);
					++index;
				}//end IF in mask
				//cout << endl;
			}//end for x
		}//end for y
		GENL_DEBUG_3(
				cout << "Pushed back " << fv.size() << " feature values with maskcnt = " << mcnt << "and fcnt = " << fcnt << endl;
		);
		features.features.push_back(fv);
		features.offsets.push_back(ov);
		features.quadUL.push_back(UL);
		features.quadUR.push_back(UR);
		features.quadLL.push_back(LL);
		features.quadLR.push_back(LR);
		features.bbox.push_back(R);
		GENL_DEBUG_2(
			cout << "At end: features.features[" << features.features.size() - 1 << "] =" << features.features[features.features.size() - 1].size() << endl;
		);
		return (int)features.features.size() - 1;
	}

