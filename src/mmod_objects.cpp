/*
 * mmod_objects.cpp
 *
 *  Created on: Sep 8, 2011
 *      Author: Gary Bradski
 */
#include "mmod_objects.h"
#include <sstream>


using namespace cv;
using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
/**
 *\brief This class stores multi-mod object models
 */

	/**
	 * \brief Empty all vectors.
	 */
	void mmod_objects::clear_matches()
	{
		if(!rv.empty())
		{
			rv.clear();
			scores.clear();
			ids.clear();
			frame_nums.clear();
			modes_used.clear();
			feature_indices.clear();
		}
	}



	/**
	 *\brief  Draw matches after a call to match_all_objects. This function is for visualization
	 * @param I   Image you want to draw onto, must be CV_8UC3. No bounds checking done
	 * @param o   Offset for drawing
	 */
	void mmod_objects::draw_matches(Mat &I, Point o)
	{
		int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
		double fontScaleS = 0.2;
		double fontScaleO = 0.3;
		int thickness = 1;
		stringstream ss;

		vector<Rect>::iterator ri;			//Rectangle iterator
		vector<float>::iterator si;			//Scores iterator
		vector<string>::iterator ii;		//Object ID iterator (object names)
		vector<vector<int> >::iterator fitr;//Feature indices iterator
		int len = (int)rv.size();
		int Dcolor = 150/len;
		Scalar color(255,255,255);
		string stringscore;
		vector<string>::iterator moditr; //Mode iterator
		int indices;
		int num_modes = (int)modes_used.size();
		if(num_modes == 0) num_modes = 1;
		int dmode = 150/num_modes;
		int i;
		for(i = 0, ri = rv.begin(), si = scores.begin(), ii = ids.begin(), fitr = feature_indices.begin(); ri != rv.end(); ++ri, ++si, ++ii, ++i, ++fitr)
		{
			color[i%3] -= Dcolor;		//Provide changing color
			Rect R(ri->x + o.x, ri->y + o.y, ri->width, ri->height);
			int cx = R.x + R.width/2, cy = R.y + R.height/2;
			rectangle(I, R, color); //Draw rectangle
			ss << *si;
			ss >> stringscore;
//			cout << "string score = " << stringscore << ", *si = " << *si << endl;
			putText(I, *ii, Point(R.x,R.y - 2), fontFace, fontScaleO, color, thickness, 8); //Object ID, and then score
			putText(I, stringscore,Point(R.x + 1, R.y + R.height/2),fontFace, fontScaleS, color, thickness, 8);
			//DRAW THE ACTUAL FEATURES THEMSELVES
//			cout << "Inner loop obj::draw_matches, ri("<<R.x<<","<<R.y<<","<<R.width<<","<<R.height<<"), si" << *si <<", ii"<<*ii<<", cxy("<<cx<<","<<cy<<")"<<endl;
//			cout << "modes_used.size() = " << modes_used.size()<<endl;
			for(indices = 0, moditr = modes_used.begin(); moditr != modes_used.end(); ++moditr, ++indices)
			{
				int matchIdx = (*fitr)[indices];
//				cout << "mode# " << indices << ", matchIdx = "<< matchIdx << endl;
				vector<Point>::iterator pitr = modes[*moditr].objs[*ii].offsets[matchIdx].begin();
				for(;pitr != modes[*moditr].objs[*ii].offsets[matchIdx].end(); ++pitr)//, ++ucharitr)
				{
					I.at<Vec3b>(pitr->y + cy,pitr->x + cx)[1] = 255 - indices*dmode; //draw in decreasing color for each mode
				}
//				cout << "And out of pixels" << endl;
			}
//			cout << "And out of modes" << endl;
		}//End for each surviving non-max suppressed feature left
//		cout << "And out of draw_matches" << endl;
	}




	/**
	 *\brief  cout all matches after a call to match_all_objects. This function is for debug
	 */
	void mmod_objects::cout_matches()
	{
		vector<Rect>::iterator ri;
		vector<float>::iterator si;
		vector<string>::iterator ii;
		string stringscore;
		int i;
		for(ri = rv.begin(), si = scores.begin(), ii = ids.begin(); ri != rv.end(); ++ri, ++si, ++ii)
		{
			cout << *ii << ": score = " << *si << " at R(" <<ri->x << ", " << ri->y << ", " << ri->width << ", " << ri->height << ")" << endl;
		}
	}



	/**
	 * \brief Find all objects above match_threshold at a point in image.
	 *
	 * Search for all matches at a point in an image over the modes (CV_8UC1 feature images) that are passed in.
	 * No checking done if vectors are empty
	 * Results are stored in rv (rect), scores (match value), objs (object_IDs) frame_nums (frame#s).
	 *
	 * @param I					For each mode, Feature image of uchar bytes where only one or zero bits are on.
	 * @param mode_names		List of names of the modes of the above features
	 * @param pp				Point at which to search
	 * @param match_threshold	Matches have to be above this score [0,1] to be considered a match
	 * @return					Number of object matches above match_threshold Results are stored in
	 * 							rv (rect), scores (match value), objs (object_IDs) frame_nums (frame#s),
	 * 							modes_used (feature types ... gradient, depth...) and feature_indices (which mmod_feature vec matched).
	 */
	int mmod_objects::match_all_objects_at_a_point(const vector<Mat> &I, const vector<string> &mode_names,
			const Point &pp, float match_threshold)
	{
		clear_matches();
		vector<Mat>::const_iterator Iit;
		vector<string>::const_iterator modit;

		//Collect matches
		vector<string> obj_names; //To be filled by return_object_names below
		vector<string>::iterator nit; //obj_names iterator
		ModelsForModes::iterator mfmit = modes.begin();
		int num_names = mfmit->second.return_object_names(obj_names);

		int match_index,frame_number;
		float score;
		float norm = (float)I.size();
		Rect R;
		//GO THROUGH EACH OBJECT
		vector<int> match_indices;
		bool collect_modes = true;
		for(nit = obj_names.begin(); nit != obj_names.end(); ++nit)
		{
			score = 0.0;
			match_indices.clear();
			//GO THROUGH EACH MODE SUMMING SCORES
			for(modit = mode_names.begin(), Iit = I.begin(); modit != mode_names.end();++Iit, ++modit)
			{
				if(modes.count(*modit)>0) //We have this mode
				{
					if(collect_modes)
						modes_used.push_back(*modit);
					score += modes[*modit].match_an_object(*nit,*Iit,pp,match_index,R,frame_number);
					match_indices.push_back(match_index);
//					objs_modal_features.push_back(modes[*modit].objs[*nit].features[match_index]);
				}
			}
			collect_modes = false;
			score /= norm;  //Normalize by number of modes
			if(score > match_threshold) //If we have a match, enter it as a contender
			{
				rv.push_back(Rect(R.x + R.width/2,R.y + R.height/2,R.width,R.height));//Our rects are middle based, make this Upper Left based
				scores.push_back(score);
				ids.push_back(*nit);
				frame_nums.push_back(frame_number);
				feature_indices.push_back(match_indices);
//				object_feature_map.insert(pair<string, vector<vector<uchar> > >(*nit,objs_modal_features));
			}
		}
		return (int)rv.size();
	}


	/**
	 * \brief Find all objects within the masked part of an image. Do non-maximum suppression on the list
	 *
	 * Search a whole image within an (optional) mask for objects, skipping (skipY,skipX) each time. Non-max suppress the result.
	 * Results are stored in rv (rect), scores (match value), objs (object_IDs) frame_nums (frame#s).
	 *
	 * @param I					For each mode, Feature image of uchar bytes where only one or zero bits are on.
	 * @param mode_names		List of names of the modes of the above features
	 * @param Mask				Mask of where to search. If empty, search the whole image. If not empty, it must be CV_8UC1 with same size as I
	 * @param match_threshold	Matches have to be above this score [0,1] to be considered a match
	 * @param frac_overlap		the fraction of overlap between 2 above threshold feature's bounding box rectangles that constitutes overlap
	 * @param skipX				In the search, jump over this many pixels X
	 * @param skipY				In the search, jump over this many pixels Y
	 * @return					Number of surviving non-max suppressed object matches.
	 * 							rv (rect), scores (match value), objs (object_IDs) frame_nums (frame#s),
	 * 							modes_used (feature types ... gradient, depth...) and feature_indices (which mmod_feature vec matched).
	 */
	int mmod_objects::match_all_objects(const vector<Mat> &I, const vector<string> &mode_names, const Mat &Mask,
			float match_threshold, float frac_overlap, int skipX, int skipY)
	{
		clear_matches();
		if(I.empty())
		{
			cerr << "ERROR, in match_all_objects, feature vector is empty." << endl;
			return -1;
		}
		vector<Mat>::const_iterator Iit;
		vector<string>::const_iterator modit;
		if(!Mask.empty())
		{
			for(Iit = I.begin(), modit = mode_names.begin(); Iit != I.end(); ++Iit, ++modit)

			{
				if(Iit->size() != Mask.size())
				{
					cerr << "ERROR in match_all_objects: I[" << *modit <<"].size.width(" << (Iit->size()).width << ") != Mask.size(" << (Mask.size()).width << ")" << endl;
					return -1;
				}
				if(Iit->type() != Mask.type())
				{
					cerr << "ERROR in match_all_objects: I[" << *modit <<"].type(" << Iit->type() << ") != Mask.type(" << Mask.type() << ")" << endl;
					return -1;
				}
			}
		}
		//Collect matches
		vector<string> obj_names; //To be filled by return_object_names below
		vector<string>::iterator nit; //obj_names iterator
		ModelsForModes::iterator mfmit = modes.begin();
		int num_names = mfmit->second.return_object_names(obj_names);

		int match_index,frame_number;
		float score;
		float norm = (float)I.size();
//		cout << "In mmod_objects::match_all_objects, norm = " << norm << endl;
		Rect R;
		vector<int> match_indices;
		bool collect_modes = true;
		if(Mask.empty()) //THERE IS NO MASK, SEARCH THE WHOLE IMAGE:
		{
//			cout << "In mask empty" << endl;

			for(int y = 0; y<I[0].rows; y += skipY)
			{
				for(int x = 0; x<I[0].cols; x += skipX)
				{
					//go through each object,
					for(nit = obj_names.begin(); nit != obj_names.end(); ++nit)
					{
						score = 0.0;
						match_indices.clear();
						//go through each mode, summing scores
						for(modit = mode_names.begin(), Iit = I.begin(); modit != mode_names.end();++Iit, ++modit)
						{
							if(modes.count(*modit)>0) //We have this mode
							{
								if(collect_modes)
									modes_used.push_back(*modit);
								Point pp = Point(x,y);
								score += modes[*modit].match_an_object(*nit,*Iit,pp,match_index,R,frame_number);
								match_indices.push_back(match_index);
							}
						}
						collect_modes = false;
						score /= norm;  //Normalize by number of modes
						if(score > match_threshold) //If we have a match, enter it as a contender
						{
							rv.push_back(Rect(R.x + x,R.y + y,R.width,R.height));//Our rects are middle based, make this Upper Left based
							scores.push_back(score);
							ids.push_back(*nit);
							frame_nums.push_back(frame_number);
							feature_indices.push_back(match_indices);
//							feature_indices.insert(feature_indices.end(),match_indices.begin(),match_indices.end());
						}
					}
				}
			}//end going over rows of the image
		}
		else //USE THE MASK:
		{
//			cout << "obj_names.size() = " << obj_names.size() <<  ", modes = " << mode_names.size() << endl;

			for(int y = 0; y<I[0].rows; y += skipY)
			{
				const uchar *m = Mask.ptr<uchar>(y);
				for(int x = 0; x<I[0].cols; x += skipX, m+=skipX)
				{
//					cout << "(" << x << ", " << y << ") m= " << (int)(*m) << endl;
					if(*m) //Mask covers this point
					{
//						if(!(y%10) && (x == y)) cout << "x,y(" << x << ", " << y << ")";// << endl;
						//go through each object,
						for(nit = obj_names.begin(); nit != obj_names.end(); ++nit)
						{
//							if(!(y%10) && (x == y)) cout << "\nvvvvv"<< *nit <<"vvvvv" << endl;
							score = 0.0;
							match_indices.clear();
							//go through each mode, summing scores
							for(modit = mode_names.begin(), Iit = I.begin(); modit != mode_names.end();++Iit, ++modit)
							{
								if(modes.count(*modit)>0) //We have this mode
								{
									if(collect_modes)
									{
										modes_used.push_back(*modit);
//										cout << "modes_used.push_back("<<*modit<<"), len"<<modes_used.size()<<endl;
									}
									Point pp = Point(x,y);
									float tscore;
									tscore = modes[*modit].match_an_object(*nit,*Iit,pp,match_index,R,frame_number);
									match_indices.push_back(match_index);
									score += tscore;
//									if(!(y%10) && (x == y)) cout << "For object " << *nit << ", mode " << *modit << ",score = " << tscore <<
//											", cumscore = " << score << " R(" << R.x <<","<<R.y<<","<<R.width<<","<<R.height<<")"<< endl;
								}
							}
							collect_modes = false;
							score /= norm;  //Normalize by number of modes
//							if(!(y%10) && (x == y)) cout << "score(" << score << ") >? match_threshold = " << match_threshold << endl;
							if(score > match_threshold) //If we have a match, enter it as a contender
							{
//								if(!(y%10) && (x == y)) cout << *nit << " Push back R(" << R.x + x<< ", " << R.y  + y<< ", " << R.width << ", " << R.height << ")" << endl;
								rv.push_back(Rect(R.x + x,R.y + y,R.width,R.height)); //Our rects are middle based, make this Upper Left based
								scores.push_back(score);
								ids.push_back(*nit);
								frame_nums.push_back(frame_number);
								feature_indices.push_back(match_indices);
							}
						}
					}
				}
			}//end going over rows of the image
		}
//		cout << "Pre nonMax, we have " << rv.size() << " potential objects" << endl;
		//Get rid of spurious overlaps:
		int num_objs = util.nonMaxRectSuppress(rv, scores, ids, frame_nums, feature_indices, frac_overlap);
//		cout << "Post nonMax, we have " << rv.size() << " potential objects" << endl;
//		cout << "____________________\n" << endl;
		return num_objs;
	}


	/**
	 * \brief Learn a template if no other template matches this view of the object well enough.
	 *
	 * This is the most common method of learning a template: Only actually learn a template if
	 * if no other template is close to the current features in the mask. So, this routine wraps the
	 * other learn_a_template and only actually records the template if no existing template scores above
	 * the learn_thresh.
	 *
	 * @param Ifeats			For each mode, Feature image of uchar bytes where only one or zero bits are on.
	 * @param mode_names		List of names of the modes of the above features
	 * @param Mask				uchar mask of the object to be learned
	 * @param framenum			Frame number of this object, so that we can reconstruct pose from the database
	 * @param learn_thresh		If no features from f match above this, learn a new template.
	 * @return					Returns total number of templates for this object
	 */
	int mmod_objects::learn_a_template(vector<Mat> &Ifeat, const vector<string> &mode_names, Mat &Mask,
			string &session_ID, string &object_ID, int framenum, float learn_thresh)
	{
		vector<Mat>::iterator Iit;
		vector<string>::const_iterator mit;
		int num_models = 0;
		for(Iit = Ifeat.begin(), mit = mode_names.begin(); Iit != Ifeat.end(); ++Iit, ++mit)
		{
//			cout << *mit << ":" << endl;
			if(modes.count(*mit)>0) //We have models already for this mode
			{
//				cout << "Have models for this mode" << endl;
				modes[*mit].learn_a_template(*Iit, Mask, session_ID, object_ID, framenum, learn_thresh);
			}
			else //We have no models for this mode yet. Better insert one
			{
//				cout << "Learning a new model for this mode" << endl;
				mmod_mode m(*mit);
				modes.insert(pair<string, mmod_mode>(*mit,m));
//				cout << "modes[*mit].mode = " << modes[*mit].mode << endl;
//				cout << "  ... learn a template with the mode. learn_thresh: " << learn_thresh << endl;
				modes[*mit].learn_a_template(*Iit, Mask, session_ID, object_ID, framenum, learn_thresh);
			}
			num_models += (int)(modes[*mit].objs[object_ID].features.size());
//			cout << "num_models = " << num_models << endl;
		}
		return num_models;
	}


