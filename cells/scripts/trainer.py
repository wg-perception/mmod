#!/usr/bin/env python
"""
Module defining the MMOD trainer to train the MMOD models
"""

from ecto_object_recognition import capture, mmod
from ecto_opencv import calib, features2d, highgui
from g2o import SbaDisparity
from feature_descriptor import FeatureDescriptor
import ecto
from object_recognition.pipelines.training import TrainingPipeline
from object_recognition.common.utils import dict_to_cpp_json_str

########################################################################################################################
class MmodModelBuilder(ecto.BlackBox):
    """
    """
    def declare_params(self, p):
        self.feature_descriptor = FeatureDescriptor()
        p.declare('visualize', 'If true, displays images at runtime', False)
        p.forward('thresh_learn', cell_name='trainer', cell_key='thresh_learn')
        p.forward('n_level_max')

    def declare_io(self, p, i, o):
        self.source = ecto.PassthroughN(items=dict(image='An image',
                                                   depth='A depth image',
                                                   mask='A mask for valid object pixels.',
                                                   K='The camera matrix',
                                                   R='The rotation matrix',
                                                   T='The translation vector',
                                                   frame_number='The frame number.'
                                                   )
                                        )
        self.trainer = mmod.MModTrainer()

        i.forward_all('source')
        o.forward_all('model_stacker')

    def configure(self, p, i, o):
        self.n_level_max = p.n_level_max
        self.pyr_depth = mmod.Pyramid(n_levels=self.n_level_max)
        self.pyr_image = mmod.Pyramid(n_levels=self.n_level_max)
        self.pyr_mask = mmod.Pyramid(n_levels=self.n_level_max)
        self.visualize = p.visualize

    def connections(self):
        graph = []
        # resize the inputs
        graph += [ self.source['image'] >> self.pyr_image['image'],
                   self.source['mask'] >> self.pyr_mask['image'],
                   self.source['depth'] >> self.pyr_depth['image']]

        # Send teh data to the trainer
        graph += [ self.source['frame_number'] >> mmod_trainer['frame_number'],
                    self.pyr_image['level_' + str(N_LEVEL_MAX-1)] >> self.trainer['image'],
                    self.pyr_mask['level_' + str(N_LEVEL_MAX-1)] >> self.trainer['mask'],
                    self.pyr_depth['level_' + str(N_LEVEL_MAX-1)] >> self.trainer['depth'],]

        if self.visualize:
            graph += [ self.pyr_image['image'] >> imshow(name='image')['image'],
                          self.pyr_mask['image'] >> imshow(name='mask')['image'],
                          self.pyr_depth['image'] >> imshow(name='depth')['image'] ]
            graph += [ mmod_trainer['filter_vis'] >> imshow(name='filter')['image'],
                        mmod_trainer['grad_vis'] >> imshow(name='grad')['image'] ]

        return graph

class MMODPostProcessor(ecto.BlackBox):
    """
    """

    def declare_params(self, p):
        pass

    def declare_io(self, p, i, o):
        pass

    def configure(self, p, i, o):
        pass

    def connections(self):
        return []

class MmodTrainingPipeline(TrainingPipeline):
    '''Implements the training pipeline functions'''

    @classmethod
    def type_name(cls):
        return "MMOD"

    def incremental_model_builder(self, pipeline_params, args):
        json_model_params = pipeline_params.get("json_model_params", False)
        if not feature_params:
            raise RuntimeError("You must supply feature_descriptor parameters for MMOD.")
        #grab visualize if works.
        visualize = getattr(args, 'visualize', False)
        return MmodModelBuilder(json_model_params=dict_to_cpp_json_str(json_model_params), visualize=visualize)

    def post_processor(self, pipeline_params, _args):
        return MMODPostProcessor()
