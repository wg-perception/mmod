#include <ecto/ecto.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//mmod inclues
#include "mmod_general.h"
#include "mmod_objects.h"
#include "mmod_mode.h"
#include "mmod_features.h"
#include "mmod_color.h"

//For serialization
#include <fstream>
#include <boost/archive/binary_oarchive.hpp>

using ecto::tendrils;
using ecto::spore;
namespace mmod
{
  struct MModPersister
  {
    static void
    declare_params(tendrils& params)
    {
      params.declare(&MModPersister::filename_objects,"filename_objects");
      params.declare(&MModPersister::filename_filter,"filename_filter");
    }
    static void
    declare_io(const tendrils& params, tendrils& in, tendrils& out)
    {
      in.declare(&MModPersister::objects_in,"objects");
      in.declare(&MModPersister::filters_in,"filters");
    }
    int process(const tendrils& /*in*/, const tendrils& /*out*/)
    {
      {
        std::ofstream filter_out(filename_filter->c_str());
		    boost::archive::binary_oarchive oa(filter_out);
		    oa << *filters_in;
      }
      {
        std::ofstream objects_out(filename_objects->c_str());
		    boost::archive::binary_oarchive oa(objects_out);
		    oa << *objects_in;
      }
      return ecto::OK;
    }
    spore<std::string> filename_filter,filename_objects;
    spore<mmod_objects> objects_in;
    spore<mmod_filters> filters_in;
  };
}

ECTO_CELL(mmod, mmod::MModPersister, "MModPersister","???");

