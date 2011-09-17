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

from mmod import MModTrainer, MModPersister

def parse_args():
    parser = argparse.ArgumentParser(description='Computes a surface mesh of an object in the database')
    parser.add_argument('-s', '--session_id', metavar='SESSION_ID', dest='session_id', type=str, default='',
                       help='The session id to reconstruct.')
    parser.add_argument('-o', '--output', metavar='OUTPUTFILE', dest='output', type=str, default='trained.mmod',
                       help='The output file of training.')
    parser.add_argument('--all', dest='compute_all', action='store_const',
                        const=True, default=False,
                        help='Compute templates for all possible sessions.')
    parser.add_argument('--visualize', dest='visualize', action='store_const',
                        const=True, default=False,
                        help='Turn on visualization.')
    object_recognition.dbtools.add_db_options(parser)
    args = parser.parse_args()
    if args.compute_all == False and args.session_id == '':
        parser.print_usage()
        print 'You must supply either a session_id or --all'
        sys.exit(1)
    return args

def train_mmod(mmod_trainer, persister, obj_ids, args):
    db_reader = capture.ObservationReader('db_reader', db_url=args.db_root, collection='observations')
    observation_dealer = ecto.Dealer(typer=db_reader.inputs.at('observation'), iterable=obj_ids)
    db_reader = capture.ObservationReader('db_reader', db_url=args.db_root, collection='observations')
    erode = imgproc.Erode(kernel=3) #-> 7x7
    rescale_depth = capture.RescaledRegisteredDepth() #this is for SXGA mode scale handling.
    plasm = ecto.Plasm()
    #connect some initial filters
    plasm.connect(
        observation_dealer[:] >> db_reader['observation'],
        db_reader['image'] >> rescale_depth['image'],
        db_reader['depth'] >> rescale_depth['depth'],
        db_reader['mask'] >> erode['image'],
    )

    print "Training session_id:", str(session.id), " object_id:", str(session.object_id)
    #set the session id, and the object id fo this guy.
    mmod_trainer.params.session_id = str(session.id)
    mmod_trainer.params.object_id = str(session.object_id)
    #hook up the trainer
    plasm.connect(
        erode['image'] >> mmod_trainer['mask'],
        db_reader['image'] >> mmod_trainer['image'],
        rescale_depth['depth'] >> mmod_trainer['depth'],
        db_reader['frame_number'] >> mmod_trainer['frame_number'],
    )
    
    #persistance
    persisert_if = ecto.If(cell=persister) #this is so it never runs.
    #only connect the If, not the persister.
    plasm.connect(mmod_trainer['templates'] >> persisert_if['templates'])

    if args.visualize:
        plasm.connect(
          db_reader['image'] >> highgui.imshow('image', name='image')[:],
          db_reader['depth'] >> highgui.imshow('depth', name='depth')[:],
          erode['image'] >> highgui.imshow('mask', name='mask')[:],
          )
    sched = ecto.schedulers.Singlethreaded(plasm)
    sched.execute()


if "__main__" == __name__:
    args = parse_args()
    #database ritual
    couch = couchdb.Server(args.db_root)
    dbs = dbtools.init_object_databases(couch)
    models.sync_models(dbs)

    sessions = dbs['sessions']
    observations = dbs['observations']
    
    mmod_trainer = MModTrainer(session_id='NA', object_id='NA')
    persister = MModPersister(filename="trained_all.mmod")

    if args.compute_all:
        results = models.Session.all(sessions)
        for session in results:
            obs_ids = models.find_all_observations_for_session(observations, session.id)
            train_mmod(mmod_trainer, persister, obs_ids, args)

    else:
        session = models.Session.load(sessions, args.session_id)
        if session == None or session.id == None:
            print "Could not load session with id:", args.session_id
            sys.exit(1)
        obs_ids = models.find_all_observations_for_session(observations, session.id)
        train_mmod(mmod_trainer, persister, obs_ids, args)
    
    #the inputs should still be valid, so just thunk process once to save to disk
    persister.process()
