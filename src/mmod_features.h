/*
 * mmod_features.h
 *
 *  Created on: Sep 8, 2011
 *      Author: Gary Bradski
 */

#ifndef MMOD_FEATURES_H_
#define MMOD_FEATURES_H_
#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
#include <vector>
//serialization
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>

namespace boost {
namespace serialization {

template<class Archive>
void serialize(Archive & ar, cv::Point &p, const unsigned int version)
{
    ar & p.x & p.y;
}
template<class Archive>
void serialize(Archive & ar, cv::Rect &r, const unsigned int version)
{
    ar & r.x & r.y & r.width & r.height;
}
} // namespace serialization
} // namespace boost

//////////////////////////////////////////////////////////////////////////////////////////////
/**
 *\brief This class stores line mode views+features-of-those-views and related structures for a specific object
 *
 * This class is mainly called via mmod_objects:: class which contains vector<mmod_features>.
 */
class mmod_features
{
    friend class boost::serialization::access;

public:
	std::string session_ID;              				//For database lookup
	std::string object_ID;								//What's the name of this object
	std::vector<int> frame_number;						//Frame number can be associated with view/pose etc
	std::vector<std::vector<uchar> > features;  				//uchar features, only one bit is on
	std::vector<std::vector<cv::Point> > offsets;   				//the x,y coordinates of each feature
	std::vector<cv::Rect>  bbox;								//bounding box of the features
	std::vector<std::vector<int> > quadUL,quadUR,quadLL,quadLR;//List of features in each quadrant
	cv::Rect max_bounds;								//This rectangle contains the maximum width and and height spanned by all the bbox rectangles
	mmod_features();

	mmod_features(std::string &sID, std::string &oID);

	/**
	 * \brief Private function to set the session and object ID
	 * @param sID
	 * @param oID
	 */
	void setup(std::string &sID, std::string &oID);


	//SERIALIZATION
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & session_ID;
        ar & object_ID;
        ar & frame_number;
        ar & features;
        ar & offsets;
        ar & bbox;
        ar & quadUL;
        ar & quadUR;
        ar & quadLL;
        ar & quadLR;
        ar & max_bounds;
    }




	/**
	 * \brief Return the overall bounding rectangle of the vector<Rect>  bbox;
	 *
	 * Return the overall bounding rectangle of the vector<Rect>  bbox; Normally you do not have to call this since it is maintained by the
	 * mmod_general::learn_a_template
	 *
	 * @return Rectangle containing the overall bounding box of the rectangle vector<Rect> bbox;
	 */
	cv::Rect find_max_template_size();


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
	int insert(mmod_features &f, int index);


};

#endif /* MMOD_FEATURES_H_ */
