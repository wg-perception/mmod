/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */
#include <ecto/ecto.hpp>
#include <string>
#include <map>
#include <vector>

#include <boost/foreach.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/flann/flann.hpp>

#include "object_recognition/common/types.h"
#include "object_recognition/db/db.h"
#include "mmod_objects.h"  //For train and test (includes mmod_mode.h, mmod_features.h, mmod_general.h
#include "mmod_color.h"    //For depth and color processing (yes, I should change the name)
namespace mmod
{
  struct TemplateLoader
  {
    static void
    declare_params(ecto::tendrils& p)
    {
      p.declare<std::string> ("collection_models", "The collection where the models are stored.").required(true);
      p.declare<object_recognition::db_future::ObjectDbParameters> ("db_params", "The DB parameters").required(true);
      p.declare<boost::python::object> ("model_ids", "The list of model ids we should consider.\n").required();
      p.declare<boost::python::object> ("object_ids", "The list of model ids we should consider.\n").required(true);
      p.declare<std::string> ("feature_descriptor_params", "JSON string describing the template parameters").required(
                                                                                                                      true);
      p.declare(&TemplateLoader::do_update_, "do_update", "Update the matcher from the database, expensive.", false);
    }

    static void
    declare_io(const ecto::tendrils& params, ecto::tendrils& inputs, ecto::tendrils& outputs)
    {
      outputs.declare<std::vector<mmod_objects> > ("templates", "The templates");
      outputs.declare<std::vector<mmod_filters> > ("filters", "The filters");
      outputs.declare<std::vector<ObjectId> > ("ids", "The matching object ids");
      outputs.declare<bool> ("do_update", "If true, that means new templates have been loaded");
    }

    void
    configure(const ecto::tendrils& params, const ecto::tendrils& inputs, const ecto::tendrils& outputs)
    {
      // Load the list of Models to study
      {
        const boost::python::object & python_model_ids = params.get<boost::python::object> ("model_ids");
        boost::python::stl_input_iterator<std::string> begin(python_model_ids), end;
        std::copy(begin, end, std::back_inserter(model_ids_));
      }

      // Load the list of Object to study
      {
        const boost::python::object & python_object_ids = params.get<boost::python::object> ("object_ids");
        boost::python::stl_input_iterator<std::string> begin(python_object_ids), end;
        std::copy(begin, end, std::back_inserter(object_ids_));
      }

      if ((model_ids_.size() != object_ids_.size()) || (model_ids_.empty()))
      {
        std::stringstream ss;
        ss << object_ids_.size() << " object ids given and " << model_ids_.size() << " model ids given." << std::endl;
        throw std::runtime_error(ss.str());
      }

      // load the descriptors from the DB
      db_params_ = params["db_params"];
      collection_models_ = params["collection_models"];

      templates_ = outputs["templates"];
      filters_ = outputs["filters"];
      do_update_out_ = outputs["do_update"];
      ids_ = outputs["ids"];
      do_update_.set_callback(boost::bind(&TemplateLoader::on_do_update, this, _1));
      *do_update_ = true;
      do_update_.dirty(true);
      do_update_.notify();
    }
    void
    on_do_update(bool on_do_update)
    {
      if (!on_do_update)
        return;

      std::cout << "Loading models. This may take some time..." << std::endl;

      *do_update_out_ = true;
      object_recognition::db_future::ObjectDb db(*db_params_);
      std::vector<ModelId>::const_iterator model_id = model_ids_.begin(), model_id_end = model_ids_.end();
      std::vector<ObjectId>::const_iterator object_id = object_ids_.begin();
      templates_->reserve(model_ids_.size());
      filters_->reserve(model_ids_.size());
      ids_->reserve(model_ids_.size());
      for (; model_id != model_id_end; ++model_id, ++object_id)
      {
        std::cout << "Loading model for object id: " << *object_id << std::endl;
        object_recognition::db_future::Document doc(db, *collection_models_, *model_id);
        mmod_objects templates;
        doc.get_attachment<mmod_objects> ("templates", templates);
        templates_->push_back(templates);

        // Store the id conversion
        ids_->push_back(*object_id);

        // Store the 3d positions
        mmod_filters filters;
        doc.get_attachment<mmod_filters> ("filters", filters);
        filters_->push_back(filters);
      }
    }

    int
    process(const ecto::tendrils& inputs, const ecto::tendrils& outputs)
    {
      *do_update_out_ = *do_update_;
      *do_update_ = false;
      return ecto::OK;
    }

    /** The collection where the models are stored */
    ecto::spore<std::string> collection_models_;
    /** The objects ids to use */
    std::vector<ObjectId> object_ids_;
    /** The matching model ids to use */
    std::vector<ModelId> model_ids_;
    /** the DB JSON parameters */
    ecto::spore<object_recognition::db_future::ObjectDbParameters> db_params_;

    /** The filters */
    ecto::spore<std::vector<mmod_filters> > filters_;
    /** The templates */
    ecto::spore<std::vector<mmod_objects> > templates_;
    /** If True, that means we got new data */
    ecto::spore<bool> do_update_out_;
    /** If True, load from the db */
    ecto::spore<bool> do_update_;
    /** Matching between an OpenCV integer ID and the ids found in the JSON */
    ecto::spore<std::vector<ObjectId> > ids_;
  };
}

ECTO_CELL(mmod, mmod::TemplateLoader, "TemplateLoader",
    "Loads templates from the DB.")
