#!/usr/bin/env python
import ecto
import sys
import couchdb
from ecto_opencv.highgui import imshow
from ecto_object_recognition import capture
from object_recognition import models, dbtools
import mmod
from ecto_object_recognition.object_recognition_db import ObjectDbParameters

N_LEVEL_MAX = 2

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train MyAlgorithm on views from the database.')
    parser.add_argument('objects', metavar='OBJECT', type=str, nargs='*',
                   help='Object ids to train.')
    parser.add_argument('--all', dest='compute_all', action='store_const',
                        const=True, default=False,
                        help='Compute meshes for all possible sessions.')
    dbtools.add_db_options(parser)
    args = parser.parse_args()
    return args

args = parse_args()
db = dbtools.init_object_databases(couchdb.Server(args.db_root))

if args.compute_all:
    results = models.Session.all(db)
    for session in results:
        args.objects.append(session.object_id)

for object_id in args.objects:
    #get a list of observation ids for a particular object id.
    obs_ids = models.find_all_observations_for_object(db, object_id)

    if not obs_ids:
        print 'No observations found for object %s.' % object_id
        continue

    plasm = ecto.Plasm()
    #the db_reader transforms observation id into a set of image,depth,mask,K,R,T
    db_params = ObjectDbParameters(dict(type=args.db_type, root=args.db_root, collection=args.db_collection))
    db_reader = capture.ObservationReader("db_reader", db_params=db_params)
    #this iterates over all of the observation ids.
    observation_dealer = ecto.Dealer(tendril=db_reader.inputs.at('observation'),
                                     iterable=obs_ids)

    plasm.connect(observation_dealer[:] >> db_reader['observation'])
    
    #allocate mmod trainer each time to keep it simple.
    mmod_trainer = mmod.MModTrainer(thresh_learn=0.97,object_id=str(object_id))
    pyr_depth = mmod.Pyramid(n_levels=N_LEVEL_MAX)
    pyr_image = mmod.Pyramid(n_levels=N_LEVEL_MAX)
    pyr_mask = mmod.Pyramid(n_levels=N_LEVEL_MAX)
    mmod_persistance_ = mmod.MModPersister(filename_filter='filter_%s.txt'%str(object_id),
                                          filename_objects='objects_%s.txt'%str(object_id)
                                          )
    mmod_model_inserter_ = mmod.ModelWriter(object_id=str(object_id), model_json_params='{"none":"none"}',
                                            db_params=db_params)

    mmod_persistance = ecto.If('Persistance',
                               cell = mmod_persistance_
                              )
    mmod_model_inserter = ecto.If('ModelInserter',
                               cell = mmod_model_inserter_
                              )
    
    #connect trainer
    plasm.connect(db_reader['frame_number'] >> mmod_trainer['frame_number'],
                  db_reader['depth'] >> pyr_depth['image'],
                  db_reader['image'] >> pyr_image['image'],
                  db_reader['mask'] >> pyr_mask['image'],
                  pyr_depth['level_' + str(N_LEVEL_MAX-1)] >> mmod_trainer['depth'],
                  pyr_image['level_' + str(N_LEVEL_MAX-1)] >> mmod_trainer['image'],
                  pyr_mask['level_' + str(N_LEVEL_MAX-1)] >> mmod_trainer['mask'],
                  )
    
    #connect training debug visualization
    plasm.connect(mmod_trainer['filter_vis'] >> imshow(name='filter')['image'],
                  mmod_trainer['grad_vis'] >> imshow(name='grad')['image']
                  )
    
    #connect persistance
    plasm.connect(mmod_trainer['filters','objects'] >> mmod_persistance['filters','objects'])
    plasm.connect(mmod_trainer['filters','objects'] >> mmod_model_inserter['filters','objects'])
    
    #visualization
    plasm.connect(db_reader['image'] >> imshow(name='image')['image'],
                  db_reader['mask'] >> imshow(name='mask')['image'],
                  db_reader['depth'] >> imshow(name='depth')['image']
                  )

    sched = ecto.schedulers.Singlethreaded(plasm)
    sched.execute()
    
    #thunk one final process on the persistance to save to disk
    mmod_persistance_.process()
    mmod_model_inserter_.process()

    #After done execution upload the resulting model to the db....

