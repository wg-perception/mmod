#!/usr/bin/env python
import ecto
import ecto_ros
import sys
import couchdb
from argparse import ArgumentParser
from ecto_opencv.highgui import imshow
from ecto_opencv import highgui
from ecto_object_recognition import capture
from object_recognition import models
import mmod
from ecto_object_recognition.object_recognition_db import ObjectDbParameters, DbDocuments
from object_recognition.common.io.source import Source
from object_recognition.common.io.sink import Sink
from object_recognition.common.io.source import Source
from object_recognition.common.utils.training_detection_args import read_arguments
from object_recognition.common.utils import json_helper

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

N_LEVEL_MAX = 2

if "__main__" == __name__:
    '''
    Run mmod testing
    '''
    plasm = ecto.Plasm()
    parser = ArgumentParser()
    argv = sys.argv

    # add arguments for the source and sink
    Sink.add_arguments(parser)

    params, args, pipeline_params, do_display, db_params, db = read_arguments(parser, argv[1:])

    model_ids = []
    object_ids = set()
    for object_id in params['object_ids']:
        for model_id in models.find_model_for_object(db, object_id, 'MMOD'):
            model_ids.append(str(model_id))
            object_ids.add(object_id)
    model_documents = DbDocuments(db_params, model_ids)

    # TODO handle this properly...
    ecto_ros.init(argv, "mmod_testing", False)#not anonymous.

    source = Source.parse_arguments(params['source'])


    sink = Sink.parse_arguments(args, db, db_params, object_ids)

    #hook up the tester
    mmod_tester = MModTester(thresh_match=0.93,color_filter_thresh=0.8,skip_x=8,skip_y=8, model_documents=model_documents)
    
    # Connect the detector to the source
    pyr_img = mmod.Pyramid(n_levels=3)
    pyr_depth = mmod.Pyramid(n_levels=3)
    for key in source.outputs.iterkeys():
        if key in mmod_tester.inputs.keys():
            if key == 'image':
                plasm.connect([ source[key] >> pyr_img['image'],
                               pyr_img['level_' + str(N_LEVEL_MAX-1)] >> mmod_tester[key] ])
            elif key == 'depth':
                plasm.connect([ source[key] >> pyr_depth['image'],
                               pyr_depth['level_' + str(N_LEVEL_MAX-1)] >> mmod_tester[key] ],
                              source[key] >> highgui.imshow('depth', name='depth')[:])

    #visualize raw data
    fps = highgui.FPSDrawer()
    plasm.connect(
          mmod_tester['debug_image'] >> highgui.imshow('mmod debug', name='mmod depth')[:],
          )
    sched = ecto.schedulers.Singlethreaded(plasm)
    sched.execute()
