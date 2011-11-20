#include <string>
#include <sstream>

#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

#include <ecto/ecto.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "object_recognition/common/json_spirit/json_spirit.h"
#include "object_recognition/common/types.h"
#include "object_recognition/db/db.h"
#include "object_recognition/db/opencv.h"
#include "object_recognition/db/ModelInserter.hpp"

#include "mmod_features.h"
#include "mmod_objects.h"

using object_recognition::db::CollectionName;
using object_recognition::db::DocumentId;

namespace mmod
{
  /** Class inserting the TOD models in the db
   */
  /** Class inserting the MMOD models in the DB
   */
  struct ModelInserterImpl: public object_recognition::db::bases::ModelInserterImpl
  {
    static void
    declare_params(ecto::tendrils& params)
    {
    }

    static void
    declare_io(const ecto::tendrils& params, ecto::tendrils& inputs, ecto::tendrils& outputs)
    {
      inputs.declare(&ModelInserterImpl::objects_, "objects", "The objects.");
      inputs.declare(&ModelInserterImpl::filters_,"filters", "The filters.");
    }

    void
    configure(const ecto::tendrils& params, const ecto::tendrils& inputs, const ecto::tendrils& outputs)
    {
    }

    std::string
    model_type() const
    {
      return "MMOD";
    }

    int
    process(const ecto::tendrils& inputs, const ecto::tendrils& outputs, object_recognition::db::Document& doc)
    {
      {
        std::stringstream objects_stream;
        boost::archive::binary_oarchive objects_archive(objects_stream);
        objects_archive << *objects_;
        doc.set_attachment_stream("objects", objects_stream);
      }
      {
        std::stringstream filters_stream;
        boost::archive::binary_oarchive filters_archive(filters_stream);
        filters_archive << *filters_;
        doc.set_attachment_stream("filters", filters_stream);
      }

      return ecto::OK;
    }
  private:
    /** The JSON parameters used to compute the model */
    ecto::spore<mmod_objects> objects_;
    ecto::spore<mmod_filters> filters_;
  };

  //for type prettiness
  typedef object_recognition::db::bases::ModelInserterBase<ModelInserterImpl> ModelInserter;
}

ECTO_CELL(mmod, mmod::ModelInserter, "ModelInserter", "Cell that saves an MMod model to the DB");
