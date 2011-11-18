#include <ecto/ecto.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/format.hpp>
using ecto::tendrils;
namespace mmod
{
    struct Pyramid
    {
        static void
        declare_params(tendrils& p)
        {
            p.declare(&Pyramid::n_levels,"n_levels","The number of pyramid levels", 2);
        }

        static void 
        declare_io(const tendrils& p, tendrils& i, tendrils& o)
        {
            int n_levels;
            p["n_levels"] >> n_levels;
            for(int iter = 0; iter < n_levels; iter++)
            {
                std::string name = boost::str( boost::format("level_%d")%iter );
                o.declare<cv::Mat>(name,"A level from the pyramid");
            }
            i.declare(&Pyramid::image,"image");
        }

        int process(const tendrils& i,const tendrils& o)
        {
            pyramid_levels.clear(); //for multithreaded apps.
            cv::buildPyramid(*image,pyramid_levels,*n_levels);
            for(int iter = 0, end = *n_levels; iter != end; iter++)
            {
                std::string name = boost::str( boost::format("level_%d")%iter );
                o[name] << pyramid_levels[iter];
            }
            return ecto::OK;
        }     
        ecto::spore<int> n_levels;
        ecto::spore<cv::Mat> image;
        std::vector<cv::Mat> pyramid_levels;
    };

}

ECTO_CELL(mmod, mmod::Pyramid, "Pyramid","???");
