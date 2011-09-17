#!/usr/bin/env python

import sys
import argparse
import time
import tempfile
import os
import math
import subprocess

import couchdb

import ecto
from ecto_opencv import calib, highgui, imgproc
import object_recognition
from object_recognition import dbtools, models, capture, observations

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

def kinect_highres(device_n=0):
    from ecto_openni import Capture, ResolutionMode, Device
    return Capture('ni device', rgb_resolution=ResolutionMode.SXGA_RES,
                   depth_resolution=ResolutionMode.VGA_RES,
                   rgb_fps=15, depth_fps=30,
                   device_number=device_n,
                   registration=True,
                   synchronize=False,
                   device=Device.KINECT
                   )
def hookup_kinect(plasm):
    '''
    returns a kinect based source of data.
    '''
    kinect_raw = kinect_highres()
    kinect_cv = highgui.NiConverter('Kinect CV')
    rescale_depth = capture.RescaledRegisteredDepth() #this is for SXGA mode scale handling.
    #connect up the kinect as input
    plasm.connect(
        kinect_raw[:] >> kinect_cv[:],
        kinect_cv['image'] >> rescale_depth['image'],
        kinect_cv['depth'] >> rescale_depth['depth'],
    )

    return (kinect_cv['image'], rescale_depth['depth'])

def hookup_db(plasm, db_root):
    couch = couchdb.Server(db_root)
    dbs = dbtools.init_object_databases(couch)
    models.sync_models(dbs)
    sessions = dbs['sessions']
    observations = dbs['observations']
    obs_ids = models.find_all_observation_ids(sessions, observations)
    db_reader = capture.ObservationReader('db_reader', db_url=db_root, collection='observations')
    observation_dealer = ecto.Dealer(typer=db_reader.inputs.at('observation'), iterable=obs_ids)
    db_reader = capture.ObservationReader('db_reader', db_url=db_root, collection='observations')
    rescale_depth = capture.RescaledRegisteredDepth() #this is for SXGA mode scale handling.
    #connect some initial filters
    plasm.connect(
        observation_dealer[:] >> db_reader['observation'],
        db_reader['image'] >> rescale_depth['image'],
        db_reader['depth'] >> rescale_depth['depth'],
    )
    return (db_reader['image'], db_reader['depth'])

def test_mmod(args):
    '''
    Run mmod testing
    '''
    plasm = ecto.Plasm()

    image , depth = None, None
    if args.use_kinect: 
        # a source of data, image, and depth, same size.  
        image, depth = hookup_kinect(plasm)
    elif args.use_db:
        image, depth = hookup_db(plasm, args.db_root)
    else:
        raise RuntimeError("No source given --use_kinect or --use_db")
    
    #hook up the tester
    mmod_tester = MModTester(filename=args.training,thresh_match=0.9,skip_x=6,skip_y=6)
    
    plasm.connect(
        image >> mmod_tester['image'],
        depth >> mmod_tester['depth'],
    )
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
