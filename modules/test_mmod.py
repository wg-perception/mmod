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
    parser = argparse.ArgumentParser(description='Computes a surface mesh of an object in the database')
    parser.add_argument('-i', '--training', metavar='TRAINING_FILE', dest='training', type=str, default='',
                       help='The training file')
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

def test_mmod(args):
    kinect_raw = kinect_highres()
    kinect_cv = highgui.NiConverter('Kinect CV')
    rescale_depth = capture.RescaledRegisteredDepth() #this is for SXGA mode scale handling.
    plasm = ecto.Plasm()
    
    #connect up the kinect as input
    plasm.connect(
        kinect_raw[:] >> kinect_cv[:],
        kinect_cv['image'] >> rescale_depth['image'],
        kinect_cv['depth'] >> rescale_depth['depth'],
    )

    #hook up the tester
    mmod_tester = MModTester(filename=args.training)
    plasm.connect(
        kinect_cv['image'] >> mmod_tester['image'],
        rescale_depth['depth'] >> mmod_tester['depth'],
    )

    #visualize raw data
    fps = highgui.FPSDrawer()
    plasm.connect(
          kinect_cv['image'] >> fps[:],
          fps[:] >> highgui.imshow('image', name='image')[:],
          kinect_cv['depth'] >> highgui.imshow('depth', name='depth')[:],
          mmod_tester['debug_image'] >> highgui.imshow('mmod debug', name='mmod depth')[:],

          )
    sched = ecto.schedulers.Singlethreaded(plasm)
    sched.execute()

if "__main__" == __name__:
    args = parse_args()
    test_mmod(args)
