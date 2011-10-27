#include <ecto/ecto.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iterator>
#include <sstream>
#include <string>

//mmod inclues
#include "mmod_general.h"
#include "mmod_objects.h"
#include "mmod_mode.h"
#include "mmod_features.h"
#include "mmod_color.h"


namespace mmod
{
//#define output_data 1

  using ecto::tendrils;
  using ecto::spore;
  using namespace std;
  struct MModTrainer
  {
#ifdef output_data
    MModTrainer()
      :
        color_filter("Color"), framenum(0), objID("foo")
    {
      modesCD.push_back("Grad");
    }
#else
    MModTrainer()
      :
        color_filter("Color")
    {
      modesCD.push_back("Grad");
    }
#endif

    static void
    declare_params(tendrils& params)
    {
      params.declare(&MModTrainer::thresh_learn,"thresh_learn","The threshold"
                                              "for learning a new template",0.0); //Zero thresh_learn => learn every view
      params.declare(&MModTrainer::object_id,"object_id",
                                             "The object id, to learn.")
                                             .required(true);

    }
    static void
    declare_io(const tendrils& params, tendrils& in, tendrils& out)
    {
      in.declare(&MModTrainer::image_in,"image", "The input image.").required(true);
      in.declare(&MModTrainer::depth_in,"depth", "The depth image.").required(true);
      in.declare(&MModTrainer::mask_in,"mask", "The mask.").required(false);
      in.declare(&MModTrainer::frame_number_in, "frame_number",
                                                " The frame number").required(true);

      out.declare(&MModTrainer::grad_vis,"grad_vis",
              "A visualization of the gradient feature.");
      out.declare(&MModTrainer::filter_vis,"filter_vis",
              "A visualization of the gradient feature.");
      out.declare(&MModTrainer::objects_out,"objects");
      out.declare(&MModTrainer::filters_out,"filters");
    }
#ifdef output_data
    string Int2string(int number)
    {
       stringstream ss;//create a stringstream
       ss << number;//add number to the stream
       return ss.str();//return a string with the contents of the stream
    }
#endif

    int process(const tendrils& /*in*/, const tendrils& /*out*/)
    {
      std::cout <<" training on image:" << *frame_number_in
                << " for object id:" << * object_id << std::endl;
#ifdef output_data
      if(objID.compare(*object_id)){ framenum = 0; objID= *object_id; cout << "Object ID changed from " << objID << " to " << *object_id << endl;}
      string fn = Int2string(framenum++);
      string jp = ".jpg";
      string png = ".png";
      string Im = "Image_"+*object_id+"_"+fn+jp, De = "Depth_"+*object_id+"_"+fn+png, Ma = "Mask_"+*object_id+"_"+fn+jp;
      cv::imwrite(Im,*image_in);
      cv::imwrite(Ma,*mask_in);
      cv::imwrite(De,*depth_in);
#endif

      //reset our outputs, so that we can be thread safe.
      *filter_vis = cv::Mat();
      *grad_vis = cv::Mat();
      if(!cv::countNonZero(*mask_in)) return ecto::OK;

      //PROCESS TO GET FEATURES
      cv::Mat colorfeat, gradfeat;
      calcHLS.computeColorHLS(*image_in,colorfeat,*mask_in,"train");
      calcGrad.computeGradients(*image_in,gradfeat,*mask_in,"train");
      g.visualize_binary_image(gradfeat, *grad_vis);
      FeatModes.clear();
      FeatModes.push_back(gradfeat);
      float Score;
      int num_templ = Objs.learn_a_template(FeatModes,modesCD, *mask_in,
	      *object_id, *object_id, *frame_number_in, *thresh_learn, &Score);
      std::cout << "#"<<*frame_number_in
          <<": Number of templates learned = " << num_templ
          <<", Score = "<<Score<< std::endl;
      int num_fs = color_filter.learn_a_template(colorfeat,*mask_in,*object_id,
                                                 *frame_number_in);
      std::cout << "Filter templates = " << num_fs << std::endl;
      g.visualize_binary_image(colorfeat,*filter_vis);
      *filters_out = color_filter;
      *objects_out = Objs;
      return ecto::OK;
    }



    spore<cv::Mat> image_in, depth_in, mask_in, grad_vis,filter_vis;
    spore<int> frame_number_in;
    spore<std::string> object_id;
    spore<float> thresh_learn;
    spore<mmod_objects> objects_out;
    spore<mmod_filters> filters_out;
    //training instances
    mmod_objects Objs;
    mmod_general g;
    mmod_filters color_filter;
    gradients calcGrad;    //Gradient feature processing
    colorhls calcHLS;

    std::vector<cv::Mat> FeatModes; //List of images
    std::vector<std::string> modesCD; //Names of modes (color and depth)
#ifdef output_data
    int framenum;
    string objID;
#endif
  };
}

ECTO_CELL(mmod, mmod::MModTrainer, "MModTrainer","???");

