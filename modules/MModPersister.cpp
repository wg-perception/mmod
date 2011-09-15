#include <ecto/ecto.hpp>
#include <opencv2/core/core.hpp>
#include <boost/archive/text_oarchive.hpp>

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
    }

    static void
    declare_io(const tendrils& p, tendrils& i, tendrils& o)
    {
      i.declare<mmod_objects>("templates");
      o.declare<std::string>("data", "Data string.");
      o.declare<std::string>("mime", "Mime type", "application/octet-stream");
    }

    void
    configure(const tendrils& p, const tendrils& i, const tendrils& o)
    {
      //outputs
      trainer_ = i["templates"];
      data_ = o["data"];
      mime_ = o["mime"];
    }

    int
    process(const tendrils& i, const tendrils& o)
    {
      std::stringstream ss;
      boost::archive::text_oarchive oa(ss);
      ss << trainer_;
      *data_ = ss.str();
      return ecto::OK;
    }
    //inputs
    spore<mmod_objects> trainer_; //Object train and test.
    //outputs
    spore<std::string> mime_, data_;
  };
}
ECTO_CELL(mmod, mmod::MModPersister, "MModPersister", "An mmod template persister.");
