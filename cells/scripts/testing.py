#!/usr/bin/env python
import ecto
import sys
import couchdb
from ecto_opencv.highgui import imshow
from ecto_object_recognition import capture
from object_recognition import models, dbtools
import mmod
from ecto_object_recognition.object_recognition_db import ObjectDbParameters
from object_recognition.common.io.source import Source

# FROM AN OLDER FILE ...
import argparse
import time
import tempfile
import os
import math
import subprocess

#import object_recognition
#from object_recognition import dbtools, models, capture, observations

from mmod import MModTester

def parse_args():
    parser = argparse.ArgumentParser(description='Test mmod features on a sequence of inputs.')
    parser.add_argument('-i', '--training', metavar='TRAINING_FILE', dest='training', type=str, default='',
                       help='The training file')
    parser.add_argument('--use_kinect', dest='use_kinect', action='store_true',
                        default=False, help='Use a highres kinect')
    parser.add_argument('--use_db', dest='use_db', action='store_true',
                        default=True, help='Use the db to test')
    dbtools.add_db_options(parser)
    args = parser.parse_args()
    if args.training == '':
        parser.print_usage()
        print 'You must supply a training file.'
        sys.exit(1)
    return args

def test_mmod(args):
    '''
    Run mmod testing
    '''
    plasm = ecto.Plasm()
    parser = ArgumentParser()

    # add arguments for the source and sink
    Sink.add_arguments(parser)

    params, args, pipeline_params, do_display, db_params, db = read_arguments(parser,argv)

    model_ids = []
    object_ids = []
    for object_id in params['object_ids']:
        for model_id in models.find_model_for_object(db, object_id, 'TOD'):
            model_ids.append(str(model_id))
            object_ids.append(object_id)
    params['object_ids'] = object_ids

    # TODO handle this properly...
    ecto_ros.init(argv, "tod_detection", False)#not anonymous.

    source = Source.parse_arguments(params['source'])


    sink = Sink.parse_arguments(args, db, db_params, params['object_ids'])

    
    
    
    #hook up the tester
    loader = mmod.TemplateLoader(collection_models=db_params.collection,
                                           db_params=db_params,
                                           object_ids=object_ids, model_ids=model_ids,
                                           feature_descriptor_params=json_helper.dict_to_cpp_json_str(pipeline_param['feature_descriptor']))

    mmod_tester = MModTester(filename=args.training,thresh_match=0.95,skip_x=8,skip_y=8)
    
    # Connect the detector to the source
    for key in source.outputs.iterkeys():
        if key in detector.inputs.keys():
            plasm.connect(source[key] >> detector[key])
    plasm.connect(loader['templates', 'objects', 'id_correspondences', 'do_update'] >>
                        detector['templates', 'objects', 'id_correspondences', 'do_update'])

    #visualize raw data
    fps = highgui.FPSDrawer()
    plasm.connect(
          image >> fps[:],
          fps[:] >> highgui.imshow('image', name='image')[:],
          depth >> highgui.imshow('depth', name='depth')[:],
          mmod_tester['debug_image'] >> highgui.imshow('mmod debug', name='mmod depth')[:],
          )
    sched = ecto.schedulers.Singlethreaded(plasm)
    sched.execute()

if "__main__" == __name__:
    args = parse_args()
    test_mmod(args)
