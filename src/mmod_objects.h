/*
 * mmod_objects.h
 *
 *  Created on: Sep 8, 2011
 *      Author: Gary Bradski
 */

#ifndef MMOD_OBJECTS_H_
#define MMOD_OBJECTS_H_
#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
#include <vector>
#include "mmod_general.h"
#include "mmod_mode.h"
//SERIALIZATION
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>

//VERBOSE
// 1 Routine list, 2 values out, 3 internal values outside of loops, 4 intenral values in loops
#define OBJS_VERBOSE 0

#if OBJS_VERBOSE >= 1
#define OBJS_DEBUG_1(X) do{X}while(false)
#else
#define OBJS_DEBUG_1(X) do{}while(false)
#endif

#if OBJS_VERBOSE >= 2
#define OBJS_DEBUG_2(X) do{X}while(false)
#else
#define OBJS_DEBUG_2(X) do{}while(false)
#endif
#if OBJS_VERBOSE >= 3
#define OBJS_DEBUG_3(X) do{X}while(false)
#else
#define OBJS_DEBUG_3(X) do{}while(false)
#endif
#if OBJS_VERBOSE >= 4
#define OBJS_DEBUG_4(X) do{X}while(false)
#else
#define OBJS_DEBUG_4(X) do{}while(false)
#endif


//////////////////////////////////////////////////////////////////////////////////////////////
/**
 *\brief This class stores multi-mod object models
 */
class mmod_objects
{
public:
	typedef std::map <std::string, mmod_mode> ModelsForModes; //(mode, models_for_that_mode)
	ModelsForModes 		modes;			//For each mode, learned objects
	mmod_general 		util;			//Learning, Matching etc (contains patch w
	//Below is just temp storage for match_all_objects* convenience
	std::vector<cv::Rect> rv;			//vector of rectangle bounding boxes from an image match_all_objects
	std::vector<float> scores;			//the scores from the above
	std::vector<std::string> ids; 		//the matched object's IDs
	std::vector<int> frame_nums;		//the matched object's frame number
	std::vector<std::string> modes_used;//Will hold the modes used for match_all_objs
	std::vector<std::vector<int> > feature_indices;	//For each object, vect of features for each mode
										//Index as follows: modes[mode name].objs[name of object].features[index of vectors]

	//SERIALIZATION
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & modes;
    }


	/**
	 *\brief  Draw matches after a call to match_all_objects. This function is for visualization
	 * @param I   Image you want to draw onto. No bounds checking done
	 * @param o   Offset for drawing
	 */
	void draw_matches(cv::Mat &I, cv::Point o = cv::Point(0,0));

	/**
	 *\brief  cout all matches after a call to match_all_objects. This function is for debug
	 */
	void cout_matches();

	/**
	 * \brief Empty all vectors.
	 */
	void clear_matches();


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
	int match_all_objects_at_a_point(const std::vector<cv::Mat> &I, const std::vector<std::string> &mode_names,
			const cv::Point &pp, float match_threshold);

	/**
	 * \brief Find all objects within the masked part of an image (it does non-maximum suppression on the list).
	 *
	 * Search a whole image within an (optional) mask for objects, skipping (skipY,skipX) each time. Non-max suppress the result.
	 * Results are stored in class members: rv (feature bounding boxes), scores (match values), objs (object_IDs) frame_nums (frame#s).
	 *
	 * @param I					Vector: for each modality, a feature image of uchar bytes where only one or zero bits are on.
	 * @param mode_names		Vector: List of names of the modes of the above features
	 * @param Mask				Mask of where to search. If empty, search the whole image. If not empty, it must be CV_8UC1 with same size as I
	 * @param match_threshold	Matches have to be above this score [0,1] to be considered a candidate match
	 * @param frac_overlap		the fraction of overlap between 2 above threshold feature's bounding box rectangles that constitutes "overlap"
	 * @param skipX				In the search, jump over this many pixels X
	 * @param skipY				In the search, jump over this many pixels Y
	 * @return					Number of surviving non-max suppressed object matches.
	 *                          Results are stored in rv (rect), scores (match value), objs (object_IDs) frame_nums (frame#s).
	 */
	int match_all_objects(const std::vector<cv::Mat> &I, const std::vector<std::string>& mode_names, const cv::Mat &Mask,
			float match_threshold, float frac_overlap, int skipX = 7, int skipY = 7);



	/**
	 * \brief Learn a template if no other template matches this view of the object well enough.
	 *
	 * Learn a template if no other template is close to the current features in the mask.
	 *
	 * @param Ifeats			Vector: For each mode, Feature image of uchar bytes where only one or zero bits are on.
	 * @param mode_names		Vector: List of names of the modes of the above features
	 * @param Mask				uchar mask silhouetting the object to be learned
	 * @param framenum			Frame number of this object, so that we can reconstruct pose from the database
	 * @param learn_thresh		If no features from f match above this, learn a new template.
	 * @return					Returns total number of templates for this object
	 */
	int learn_a_template(std::vector<cv::Mat> &Ifeat, const std::vector<std::string> &mode_names, cv::Mat &Mask,
			std::string &session_ID, std::string &object_ID, int framenum, float learn_thresh);

};

#endif /* MMOD_OBJECTS_H_ */
