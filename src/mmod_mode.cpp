/*
 * mmod_mode.cpp
 *
 *  Created on: Sep 8, 2011
 *      Author: Gary Bradski
 */
#include "mmod_mode.h"
using namespace cv;
using namespace std;
//////////////////////////////////////////////////////////////////////////////////////////////
mmod_mode::mmod_mode()
	{
		mode = "NotSet";
		patch = Mat::zeros(100,100,CV_8UC1); //patch will resize as necessary, just start it at some value
	}
mmod_mode::mmod_mode(const string &mode_name)
	{
		mode = mode_name;
		patch = Mat::zeros(100,100,CV_8UC1); //patch will resize as necessary, just start it at some value
	}
//	/**
//	 * \brief Empty all vectors.
//	 */
//	void clear_matches()
//	{
//		if(!rv.empty())
//		{
//			rv.clear();
//			scores.clear();
//			objs.clear();
//			frame_nums.clear();
//		}
//	}
	/**
	 * \brief Return all the object names in mmod_mode::objs.
	 *
	 * @param obj_names  fill this vector with names
	 * @return	Number of names
	 */
	int mmod_mode::return_object_names(vector<string> &obj_names)
	{
		obj_names.clear();
		ObjectModels::iterator o;
		int i = 0;
		for(o = objs.begin(); o != objs.end(); ++o, ++i)
		{
			obj_names.push_back(o->second.object_ID);
		}
		return i;
	}


	/**
	 * \brief Learn a template if no other template matches this view of the object well enough.
	 *
	 * This is the most common method of learning a template: Only actually learn a template if
	 * if no other template is close to the current features in the mask. So, this routine wraps the
	 * other learn_a_template and only actually records the template if no existing template scores above
	 * the learn_thresh.
	 *
	 * @param Ifeat				Feature image of uchar bytes where only one or zero bits are on.
	 * @param Mask				uchar mask of the object to be learned
	 * @param session_ID		Session ID to store if we learn a template
	 * @param object_ID			Object name to store if we learn a template
	 * @param framenum			Frame number of this object, so that we can reconstruct pose from the database
	 * @param learn_thresh		If no features from f match above this, learn a new template.
	 * @return					Returns index of newly learned template, or -1 if a template already covered
	 */
	int mmod_mode::learn_a_template(Mat &Ifeat, Mat &Mask, string &session_ID, string &object_ID,
			int framenum, float learn_thresh)
	{
//		cout << "In mmod_mode::learn_a_template" << endl;
		mmod_features ftemp(session_ID, object_ID);  //We'll learn a provisional feature here
		int index = util.learn_a_template(Ifeat, Mask, framenum, ftemp);
//		cout << "index = " << index << ", learned util.learn_a_template" << endl;
//		cout << "ftemp max_bounds:" << ftemp.max_bounds.x << ", " << ftemp.max_bounds.y << ", "<< ftemp.max_bounds.width << ", "<< ftemp.max_bounds.height << endl;
//		cout << "ftemp.bbox(" << ftemp.bbox[index].x << ", " << ftemp.bbox[index].y << ", " << ftemp.bbox[index].width << ", " <<  ftemp.bbox[index].height << ")" << endl;
		//SEE IF PATCH NEEDS TO BE ADJUSTED
		if((patch.empty())||(ftemp.bbox[index].width >= patch.cols)||(ftemp.bbox[index].height >= patch.rows))
		{
			patch = Mat::zeros(ftemp.bbox[index].height+20,ftemp.bbox[index].width+20,CV_8UC1);
		}
		int rows = patch.rows; int cols = patch.cols;
//		cout << "Patch(rows,cols)" << rows << ", " << cols << endl;
		//MAP THE FEATURE TO AN IMAGE PATCH
		int xc = cols/2, yc = rows/2;
		util.display_feature(patch, ftemp.features[index], ftemp.offsets[index], ftemp.bbox[index]);
		int match_index, object_index;
		if(objs.count(object_ID)>0) //If this object exits already
		{
			Point pp = Point(xc,yc);
//			cout << "Obj exists already, point = (" << pp.x << ", " << pp.y << ")" << endl;
			float score = util.match_a_patch_bruteforce(patch, pp, objs[object_ID], match_index);
//			cout << object_ID << " at match_index = " << match_index << ", score from bfm = " << score << endl;
			if(score <= learn_thresh)
			{
//				cout << "Return insert " << endl;
				return(objs[object_ID].insert(ftemp,index)); //Insert this template because no existing template scored it high enough
			}
		}
		else // We have never learned this object yet
		{
//			cout << "inserting new obj" << endl;
			objs.insert(pair<string, mmod_features>(object_ID,ftemp));
			return 0;//Template is in zero'th position in objs[object_ID].features
		}
		return -1;
	}


	/**
	 * \brief Return the match found at a point in the image
	 *
	 * @param object_ID		object name
	 * @param I				Feature image of uchar bytes where only one or zero bits are on.
	 * @param pp			Point to perform the match at
	 * @param match_index	The index of the match will be returned here
	 * @param R				if a match, return boundind box of the feature, else leave alone
	 * @param frame_numb	if a match, return frame_number of feature, else leave alone
	 * @return				Score of this match
	 */
	float mmod_mode::match_an_object(string &object_ID, const Mat &I, const Point &pp, int &match_index,
			Rect &R, int &frame_numb)
	{
		float score = 0.0;
		if(objs.count(object_ID)>0) //If this object exits already
		{
			score = util.match_a_patch_bruteforce(I,pp,objs[object_ID],match_index);
			R = objs[object_ID].bbox[match_index]; //This is the bounding box of the mask. It needs to be offset by pp:
//			cout << "Hey! score = " << score << " match_index = " << match_index << endl;
//			if(match_index >= 0) cout << "score = " << score << "at pp = ("<<pp.x<<","<<pp.y<<") mode::match_an_object: R(" << R.x <<","<<R.y<<","<<R.width<<","<<R.height<<")"<< endl;
//			R.x += pp.x; R.y += pp.y; //R.x and R.y were set to the center of the object
//			if(match_index >= 0) cout << "After: mode::match_an_object: R(" << R.x <<","<<R.y<<","<<R.width<<","<<R.height<<")"<< endl;
			frame_numb = objs[object_ID].frame_number[match_index];
			return(score);
		}
		//If we can't fill in R and frame_numb, don't touch them
		match_index = -1;
		return score;
	}
