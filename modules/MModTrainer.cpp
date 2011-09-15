#include <ecto/ecto.hpp>
#include <opencv2/core/core.hpp>
#include "mmod_objects.h"  //For train and test
#include "mmod_color.h"    //For depth and color processing (yes, I should change the name)
namespace mmod
{
  using ecto::tendrils;
  using ecto::spore;
  struct MModTrainer
  {
    static void
    declare_params(tendrils& p)
    {
      p.declare<float>("thresh_learn", "The threshold for learning a new template", 0.8);
      p.declare<float>("thresh_match", "The threshold for learning a new template", 0.85);
      p.declare<float>(
          "frac_overlap",
          "the fraction of overlap between 2 above threshold feature's bounding box rectangles that constitutes 'overlap'",
          0.6);
      p.declare<std::string>("session_id", "The id of the training session.").required(true);
      p.declare<std::string>("object_id", "The id of the object in the training session.").required(true);
    }

    static void
    declare_io(const tendrils& p, tendrils& i, tendrils& o)
    {
      i.declare<cv::Mat>("image", "An image. BGR image of type CV_8UC3").required(true);
      i.declare<cv::Mat>("mask", "Object mask of type CV_8UC1 or CV_8UC3").required(true);
      i.declare<cv::Mat>("depth", "Depth image of type CV_16UC1").required(true);
      i.declare<int>("frame_number", "A frame number, expected to be increasing at least monotonically.").required(
          true);
      o.declare<mmod_objects>("templates");
    }

    void
    configure(const tendrils& p, const tendrils& i, const tendrils& o)
    {
      //parameter spores
      thresh_learn_ = p["thresh_learn"];
      thresh_match_ = p["thresh_match"];
      frac_overlap_ = p["frac_overlap"];
      session_id_ = p["session_id"];
      object_id_ = p["object_id"];

      // inputs
      image_ = i["image"];
      mask_ = i["mask"];
      depth_ = i["depth"];
      frame_number_ = i["frame_number"];

      //outputs
      trainer_ = o["templates"];
      //SET UP:
      //Set up our modes (right now we have color and depth. Lets say we use that order: Color and Depth)
      modesCD.push_back("Color");
      modesCD.push_back("Depth");
    }

    int
    process(const tendrils& i, const tendrils& o)
    {
      cv::Mat colorfeat, depthfeat; //To hold feature outputs. These will be CV_8UC1 images
      //PROCESS TO GET FEATURES
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
      calcColor.computeColorWTA(*image_, colorfeat, *mask_);
      calcDepth.computeDepthWTA(depth, depthfeat, *mask_);
      FeatModes.clear();
      FeatModes.push_back(colorfeat);
      FeatModes.push_back(depthfeat);

      //LEARN A TEMPLATE (for now, it will slow down with each view learned).
      std::string sT1("ST1"), oT("T"), oX("X");
      /*int num_templ = */
      trainer_->learn_a_template(FeatModes, modesCD, *mask_, *session_id_, *object_id_, *frame_number_, *thresh_learn_);
      return ecto::OK;
    }

    colorwta calcColor; //Feature processing
    depthwta calcDepth; //Feature processing
    std::vector<cv::Mat> FeatModes; //List of images
    std::vector<std::string> modesCD; //Names of modes (color and depth)

    //parameters (dynamic)
    spore<std::string> session_id_, object_id_;
    spore<float> thresh_learn_, thresh_match_, frac_overlap_;
    //inputs
    spore<cv::Mat> image_, mask_, depth_;
    spore<int> frame_number_;
    //outputs
    spore<mmod_objects> trainer_; //Object train and test.
  };
}
ECTO_CELL(mmod, mmod::MModTrainer, "MModTrainer", "An mmod template trainer.");
