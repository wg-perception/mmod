#include <ecto/ecto.hpp>
#include <opencv2/core/core.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <fstream>

#include "mmod_objects.h"  //For train and test
#include "mmod_color.h"    //For depth and color processing (yes, I should change the name)
namespace mmod
{
  using ecto::tendrils;
  using ecto::spore;
  struct MModPersister
  {
    static void
    declare_params(tendrils& p)
    {
      p.declare<std::string> ("filename", "Output file name to save training to.");
    }

    static void
    declare_io(const tendrils& p, tendrils& i, tendrils& o)
    {
      i.declare<mmod_objects> ("templates");

      //TODO output the serialized data.
      //      o.declare<std::string>("data", "Data string.");
      //      o.declare<std::string>("mime", "Mime type", "application/octet-stream");
    }

    void
    configure(const tendrils& p, const tendrils& i, const tendrils& o)
    {
      //outputs
      templates_ = i["templates"];
      //TODO outputs
      //      data_ = o["data"];
      //      mime_ = o["mime"];
      //parameters
      p["filename"] >> filename_;
    }

    int
    process(const tendrils& i, const tendrils& o)
    {
      //serialize file to disk for now.
      std::ofstream file(filename_.c_str());
      boost::archive::text_oarchive oa(file);
      oa << *templates_;
      return ecto::OK;
    }
    //inputs
    spore<mmod_objects> templates_; //Object train and test.
    //outputs TODO
    //    spore<std::string> mime_, data_;

    std::string filename_;
  };
}
ECTO_CELL(mmod, mmod::MModPersister, "MModPersister", "An mmod template persister.")
;
