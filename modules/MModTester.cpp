#include <ecto/ecto.hpp>
#include <opencv2/core/core.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>

#include "mmod_objects.h"  //For train and test
#include "mmod_color.h"    //For depth and color processing (yes, I should change the name)
namespace mmod
{
  using ecto::tendrils;
  using ecto::spore;
  struct MModTester
  {
    static void
    declare_params(tendrils& p)
    {
      p.declare<std::string> ("filename", "Output file name to save training to.");

      p.declare<float> ("thresh_learn", "The threshold for learning a new template", 0.8);
      p.declare<float> ("thresh_match", "The threshold for learning a new template", 0.85);
      p.declare<float> (
                        "frac_overlap",
                        "the fraction of overlap between 2 above threshold feature's bounding box rectangles that constitutes 'overlap'",
                        0.6);
      p.declare<int> ("skip_x", "Control sparse testing of the feature images", 2);
      p.declare<int> ("skip_y", "Control sparse testing of the feature images", 2);
    }

    static void
    declare_io(const tendrils& p, tendrils& i, tendrils& o)
    {
      i.declare<cv::Mat> ("image", "An image. BGR image of type CV_8UC3").required(true);
      i.declare<cv::Mat> ("depth", "Depth image of type CV_16UC1").required(true);
      i.declare<cv::Mat> ("mask", "Object mask of type CV_8UC1 or CV_8UC3").required(false);
      o.declare<cv::Mat> ("debug_image", "Debug image.");
    }

    void
    configure(const tendrils& p, const tendrils& i, const tendrils& o)
    {
      std::string filename;
      //parameters
      p["filename"] >> filename;
      thresh_match_ = p["thresh_match"];
      frac_overlap_ = p["frac_overlap"];
      skip_x_ = p["skip_x"];
      skip_y_ = p["skip_y"];

      //deserialize from file.
      std::ifstream file(filename.c_str());
      boost::archive::text_iarchive ia(file);
      ia >> templates_;

      // inputs
      image_ = i["image"];
      mask_ = i["mask"];
      depth_ = i["depth"];

      //outputs
      debug_image_ = o["debug_image"];
    }

    int
    process(const tendrils& i, const tendrils& o)
    {
      //iputs spores are like smart pointers, dereference to get at under
      //lying data type.
      cv::Mat image = *image_, mask = *mask_;

      cv::Mat depth;
      if (depth_->type() == CV_32FC1)
      {
        depth_->convertTo(depth, CV_16U, 1 / 1000.0);
      }
      else if (depth_->type() == CV_16UC1)
      {
        depth = *depth_;
      }
      else
      {
        throw std::logic_error(
                               "You must supply us with either a CV_32FC1 or CV_16UC1 depth map. Floating point in meters, fixed in mm.");
      }
      //run detections.
      //TEST (note that you can also match_all_objects_at_a_point(...):
      calcColor.computeColorWTA(image, colorfeat, mask);
      calcDepth.computeDepthWTA(depth, depthfeat, mask);
      FeatModes.clear();
      FeatModes.push_back(colorfeat);
      FeatModes.push_back(depthfeat);

      templates_.match_all_objects(FeatModes, modesCD, mask, *thresh_match_, *frac_overlap_, *skip_x_, *skip_y_);

      //TO DISPLAY MATCHES (NON-MAX SUPPRESSED)
      cv::Mat debug_image;
      image.copyTo(debug_image);
      templates_.draw_matches(debug_image); //draw results...

      *debug_image_ = debug_image;
      return ecto::OK;
    }

    colorwta calcColor; //Feature processing
    depthwta calcDepth; //Feature processing
    std::vector<cv::Mat> FeatModes; //List of images
    std::vector<std::string> modesCD; //Names of modes (color and depth)
    cv::Mat colorfeat, depthfeat; //To hold feature outputs. These will be CV_8UC1 images

    mmod_objects templates_; //Object train and test.

    //params
    spore<float> thresh_match_, frac_overlap_;
    spore<int> skip_x_, skip_y_;

    //inputs
    spore<cv::Mat> image_, mask_, depth_;

    //outputs
    spore<cv::Mat> debug_image_;
  };
}
ECTO_CELL(mmod, mmod::MModTester, "MModTester", "An mmod template detector.")
;
