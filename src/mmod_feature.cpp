/*
 * mmod_feature.cpp
 *
 *  Created on: Sep 8, 2011
 *      Author: Gary Bradski
 */
#include "mmod_features.h"
using namespace cv;
using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
/**
 *\brief This class stores line mode views+features-of-those-views and related structures for a specific object
 *
 * This class is mainly called via mmod_objects:: class which contains vector<mmod_features>.
 */
mmod_features::mmod_features()
	{
		string sID("No_sID");
		string oID("No_oID");
		setup(sID,oID);
	}

mmod_features::mmod_features(string &sID, string &oID)
	{
		setup(sID,oID);
	}
	/**
	 * \brief Private function to set the session and object ID
	 * @param sID
	 * @param oID
	 */
	void mmod_features::setup(string &sID, string &oID)
	{
		session_ID = sID;
		object_ID = oID;
		max_bounds.width = -1;
		max_bounds.height = -1;
	}

	/**
	 * \brief Return the overall bounding rectangle of the vector<Rect>  bbox;
	 *
	 * Return the overall bounding rectangle of the vector<Rect>  bbox; Normally you do not have to call this since it is maintained by the
	 * mmod_general::learn_a_template
	 *
	 * @return Rectangle containing the overall bounding box of the rectangle vector<Rect> bbox;
	 */
	Rect mmod_features::find_max_template_size()
	{
		vector<Rect>::iterator rit;
		for(rit = bbox.begin(); rit != bbox.end(); ++rit)
		{
			if((*rit).width > max_bounds.width)
				max_bounds.width = (*rit).width;
			if((*rit).height > max_bounds.height)
				max_bounds.height = (*rit).height;
		}
		max_bounds.x = -max_bounds.width/2;
		max_bounds.y = -max_bounds.height/2;
		return max_bounds;
	}

	/**
	 * \brief insert a feature at a given index from an external mmod_features class into this mmod_features class
	 *
	 * We use this when we have learned a new template in mmod_general.learn_a_template, have decided to include it
	 * because no existing template matches it well (found by using mmod_general::display_feature and then
	 * mmod_general::match_a_patch_bruteforce or scalable matching).
	 *
	 * @param f			refererence to mmod_features containing 1 or more features
	 * @param index		the index of which feature we want inserted here from f above
	 * @return			the index into which the indexed value from f was inserted. -1 => error
	 */
	int mmod_features::insert(mmod_features &f, int index)
	{
		int size = (int)f.features.size();
		if(index >= size)
		{
			cerr << "ERROR, in mmod_features.insert, index = " << index << " was >= to size(" << size << ") of passed in mmod_features" << endl;
			return -1;
		}
		frame_number.push_back(f.frame_number[index]);
		features.push_back(f.features[index]);
		offsets.push_back(f.offsets[index]);
		bbox.push_back(f.bbox[index]);
		quadUL.push_back(f.quadUL[index]);
		quadUR.push_back(f.quadUR[index]);
		quadLR.push_back(f.quadLR[index]);
		quadLL.push_back(f.quadLL[index]);
		Rect R = f.bbox[index];
		if(R.width > max_bounds.width) max_bounds.width = R.width;
		if(R.height > max_bounds.height) max_bounds.height = R.height;
		max_bounds.x = -max_bounds.width/2;
		max_bounds.y = -max_bounds.height/2;
		return ((int)features.size() - 1);
	}
