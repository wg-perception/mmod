#include <fstream>

#include <boost/foreach.hpp>

#include <ecto/ecto.hpp>
#include <opencv2/core/core.hpp>

#include "object_recognition/common/types.h"
#include "object_recognition/db/db.h"
#include "object_recognition/db/ModelReader.h"

#include "mmod_objects.h"  //For train and test (includes mmod_mode.h, mmod_features.h, mmod_general.h
#include "mmod_color.h"    //For depth and color processing (yes, I should change the name)
using namespace std;
using namespace cv;
using object_recognition::db::Documents;
using object_recognition::db::ObjectId;
using object_recognition::db::ModelId;

namespace mmod
{
  using ecto::tendrils;
  using ecto::spore;
  struct MModTester : public object_recognition::db::bases::ModelReaderImpl
  {
    void
    ParameterCallback(const Documents & db_documents)
    {
      templates_.reserve(db_documents.size());
      filters_.reserve(db_documents.size());
      object_ids_.reserve(db_documents.size());

      // Re-load the data from the DB
      std::cout << "Loading models. This may take some time..." << std::endl;
      BOOST_FOREACH (const object_recognition::db::Document & document, db_documents)
          {
            ObjectId object_id = document.get_value<std::string>("object_id");
            std::cout << "Loading model for object id: " << object_id << std::endl;
            mmod_objects templates;
            document.get_attachment<mmod_objects>("objects", templates);
            templates_.push_back(templates);

            // Store the id conversion
            object_ids_.push_back(object_id);

            // Store the 3d positions
            mmod_filters filters;
            document.get_attachment<mmod_filters>("filters", filters);
            filters_.push_back(filters);
          }
    }

    static void
    declare_params(tendrils& p)
    {
      p.declare<float>("thresh_match", "The threshold for declaring an object detected", 0.95);
      p.declare<float>(
          "frac_overlap",
          "the fraction of overlap between 2 above threshold feature's bounding box rectangles that constitutes 'overlap'",
          0.6);
      p.declare<float>("color_filter_thresh", "The color filter threshold to confirm a match", 0.91);
      p.declare<int>("skip_x", "Control sparse testing of the feature images", 8);
      p.declare<int>("skip_y", "Control sparse testing of the feature images", 8);
    }

    static void
    declare_io(const tendrils& p, tendrils& i, tendrils& o)
    {
      i.declare<std::vector<ObjectId> >("ids", "The matching object ids");

      i.declare(&MModTester::image_, "image", "An image. BGR image of type CV_8UC3").required(true);
      i.declare(&MModTester::depth_, "depth", "Depth image of type CV_16UC1").required(true);
      //      i.declare<cv::Mat> ("mask", "Object mask of type CV_8UC1 or CV_8UC3").required(false);
      o.declare(&MModTester::debug_image_, "debug_image", "Debug image.");
    }

    void
    configure(const tendrils& p, const tendrils& i, const tendrils& o)
    {
      std::string filename;
      //parameters
      thresh_match_ = p["thresh_match"];
      frac_overlap_ = p["frac_overlap"];
      color_filter_thresh_ = p["color_filter_thresh"];
      skip_x_ = p["skip_x"];
      skip_y_ = p["skip_y"];
      modesCD.push_back("Grad");
      //      modesCD.push_back("Color");
      //      modesCD.push_back("Depth");
      //deserialize from file.
    }

    int
    process(const tendrils& i, const tendrils& o)
    {
      //iputs spores are like smart pointers, dereference to get at under
      //lying data type.
      cv::Mat image = *image_, depth = *depth_; // , mask = *mask_; //We don't need mask for rec

      //run detections.
      //TEST (note that you can also match_all_objects_at_a_point(...):
      calcHLS.computeColorHLS(image, colorfeat, noMask);
      calcGrad.computeGradients(image, gradfeat, noMask);

      FeatModes.clear();
      FeatModes.push_back(gradfeat);
      //      FeatModes.push_back(colorfeat);
      //      FeatModes.push_back(depthfeat);
      for (unsigned int i = 0; i < templates_.size(); ++i)
      {
        mmod_objects & mmod_object = templates_[i];
        mmod_filters & mmod_filter = filters_[i];

        int numrawmatches = 0; //Number of matches before non-max suppression
        cout << "num_matches = " << numrawmatches << endl;
        int num_matches = mmod_object.match_all_objects(FeatModes, modesCD, noMask, *thresh_match_, *frac_overlap_,
                                                        *skip_x_, *skip_y_, &numrawmatches);
        cout << "num_matches = " << num_matches << ", selected from # of raw matches = " << numrawmatches << endl;
        //      vector<float> scs = templates_.scores; //Copy the scores over

        //FILTER RECOGNITIONS BY COLOR
        mmod_filter.filter_object_recognitions(colorfeat, mmod_object, *color_filter_thresh_);

      }

      //TO DISPLAY MATCHES (NON-MAX SUPPRESSED)
      cv::Mat debug_image;
      image.copyTo(debug_image);
      for (unsigned int i = 0; i < templates_.size(); ++i)
      {
        mmod_objects & mmod_object = templates_[i];

        mmod_object.draw_matches(debug_image); //draw results...
        mmod_object.cout_matches();
      }

      *debug_image_ = debug_image;
      return ecto::OK;
    }

    colorhls calcHLS; //Color feature processing
    //  depthgrad  calcDepth; //Depth feature processing
    gradients calcGrad; //Gradient feature processing
    std::vector<cv::Mat> FeatModes; //List of images
    std::vector<std::string> modesCD; //Names of modes (color and depth)
    cv::Mat gradfeat, colorfeat, depthfeat; //To hold feature outputs. These will be CV_8UC1 images
    cv::Mat noMask; //This is simply an empty image which means to search the whole test image

    //params
    spore<float> thresh_match_, frac_overlap_, color_filter_thresh_;
    spore<int> skip_x_, skip_y_;

    //inputs
    spore<cv::Mat> image_, mask_, depth_;

    //outputs
    spore<cv::Mat> debug_image_;

    /** The filters */
    std::vector<mmod_filters> filters_;
    /** The templates */
    std::vector<mmod_objects> templates_;
    /** Matching between an OpenCV integer ID and the ids found in the JSON */
    std::vector<ObjectId> object_ids_;
  };
}
ECTO_CELL(mmod, object_recognition::db::bases::ModelReaderBase<mmod::MModTester>, "MModTester", "An mmod template detector.");
