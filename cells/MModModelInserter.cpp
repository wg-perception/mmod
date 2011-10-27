#include <string>
#include <sstream>

#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

#include <ecto/ecto.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "object_recognition/db/db.h"
#include "object_recognition/db/opencv.h"

#include "mmod_features.h"
#include "mmod_objects.h"

using object_recognition::db_future::CollectionName;
using object_recognition::db_future::DocumentId;

namespace mmod
{
  /** Class inserting the TOD models in the db
   */
  struct ModelInserter
  {
    static void
    declare_params(ecto::tendrils& params)
    {
      params.declare<std::string> ("collection_models",
                                   "std::string The collection in which to store the models on the db", "models").required(
                                                                                                                           true);
      params.declare<object_recognition::db_future::ObjectDbParameters> ("db_params", "The DB parameters").required(true);
      params.declare<std::string> ("object_id", "The object id, to associate this frame with.").required(true);
      params.declare<std::string> ("model_json_params", "The parameters used for the model, as JSON.").required(true);
    }

    static void
    declare_io(const ecto::tendrils& params, ecto::tendrils& inputs, ecto::tendrils& outputs)
    {
      inputs.declare<mmod_objects> ("objects", "The objects.");
      inputs.declare<mmod_filters> ("filters", "The filters.");
    }

    void
    configure(const ecto::tendrils& params, const ecto::tendrils& inputs, const ecto::tendrils& outputs)
    {
      object_id_ = params["object_id"];
      db_.set_params(params.get<object_recognition::db_future::ObjectDbParameters> ("db_params"));
      collection_models_ = params.get<std::string> ("collection_models");
      params_ = params.get<std::string> ("model_json_params");

      objects_ = inputs["objects"];
      filters_ = inputs["filters"];
    }

    int
    process(const ecto::tendrils& inputs, const ecto::tendrils& outputs)
    {
      object_recognition::db_future::Document doc(db_, collection_models_);

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
      doc.set_value("object_id", *object_id_);

      // Convert the parameters to a property tree and insert them
      boost::property_tree::ptree params;
      std::stringstream ssparams;
      ssparams << params_;
      boost::property_tree::read_json(ssparams, params);
      if (params.find("type") != params.not_found())
        params.erase("type");
      doc.set_values("parameters", params);

      doc.set_value("Type", "Model");
      doc.set_value("ModelType", "LINEMOD");
      std::cout << "Persisting" << std::endl;
      doc.Persist();

      return ecto::OK;
    }
  private:
    object_recognition::db_future::ObjectDb db_;
    ecto::spore<DocumentId> object_id_;
    CollectionName collection_models_;
    /** The JSON parameters used to compute the model */
    std::string params_;
    ecto::spore<mmod_objects> objects_;
    ecto::spore<mmod_filters> filters_;
  };
}

ECTO_CELL(mmod, mmod::ModelInserter, "ModelInserter","???")
;
