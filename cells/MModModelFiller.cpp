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
  struct ModelWriter
  {
    static void
    declare_io(const ecto::tendrils& params, ecto::tendrils& inputs, ecto::tendrils& outputs)
    {
      inputs.declare(&ModelWriter::objects_, "objects", "The objects.");
      inputs.declare(&ModelWriter::filters_, "filters", "The filters.");
      outputs.declare(&ModelWriter::db_document_, "db_document", "The filled document.");
    }

    int
    process(const ecto::tendrils& inputs, const ecto::tendrils& outputs)
    {
      {
        std::stringstream objects_stream;
        boost::archive::binary_oarchive objects_archive(objects_stream);
        objects_archive << *objects_;
        db_document_->set_attachment_stream("objects", objects_stream);
      }
      {
        std::stringstream filters_stream;
        boost::archive::binary_oarchive filters_archive(filters_stream);
        filters_archive << *filters_;
        db_document_->set_attachment_stream("filters", filters_stream);
      }

      return ecto::OK;
    }
  private:
    /** The JSON parameters used to compute the model */
    ecto::spore<mmod_objects> objects_;
    ecto::spore<mmod_filters> filters_;
    ecto::spore<object_recognition::db::Document> db_document_;
  };
}

ECTO_CELL(mmod, mmod::ModelWriter, "ModelWriter", "Cell that saves an MMod model to the DB");
