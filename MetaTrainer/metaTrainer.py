import tensorflow as tf
import abc
from collections import OrderedDict

import Nets
from Data_utils import preprocessing
from Losses import loss_factory
import numpy as np

class MetaTrainer(object):
    __metaclass__ = abc.ABCMeta

    """
    Abstract Class for all the meta training algorithms
    """

     #=======================Static Class Fields=============
    _valid_args = [
        ("inputs", "inputs for the model it should be a batch of tasks, each task is a video sequence"),
        ("model", "name of the model to build"),
        ("loss", "lambda to compute loss given object and inputs"),
        ("session", "session object to manipulate graph values"),
        ("alpha", "learning rate for the inner optimization loop"),
        ("lr", "learning rate")
    ]
    _learner_name="MetaTrainer"
    #=====================Static Class Methods==============

    @classmethod
    def _get_possible_args(cls):
        return cls._valid_args
    
    #==================PRIVATE METHODS======================
    def __init__(self, **kwargs):
        print('=' * 50)
        print('Starting Creation of {}'.format(self._learner_name))
        print('=' * 50)
        self._ready=False
        self._summary_ready=False

        self._validate_args(kwargs)
        print('Args Validated, setting up trainer')

        self._build_trainer(kwargs)
        self._build_updater(kwargs)
        print('Trainer set up')

        #fetch potential update ops
        self._update_ops = tf.group(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

        #create placeholder for summary_ops
        self._summary_ops=[]
    
    def _merge_summaries(self):
        self._summary_op = tf.summary.merge(self._summary_ops)

    def _build_forward(self, input_left, input_right, weight_collection, is_training=True, args_bn=False, adap_mod_pre=False, adap_mod_post=False, adap_mod_deconv=False):
        net_args = {}
        net_args['left_img'] = input_left
        net_args['right_img'] = input_right
        net_args['variable_collection'] = weight_collection
        net_args['is_training'] = is_training
        net_args['bn_apply'] = args_bn
        net_args['adap_mod_pre'] = adap_mod_pre
        net_args['adap_mod_post'] = adap_mod_post
        net_args['adap_mod_deconv'] = adap_mod_deconv

        # MADNet
        net_args['split_layers'] = [None]
        net_args['sequence'] = True
        net_args['train_portion'] = 'BEGIN'
        net_args['bulkhead'] = False
        return Nets.factory.getStereoNet(self._model, net_args) 
    
    def _check_for_ready(self):
        if not self._ready:
            raise Exception("You should not be here")
    
    def _build_updated_variables(self,names,variables,gradients,args):
        """
        Create a new dictionary where for each name in names there is a copy of a variable in variables updated according toits gradient in gradients 
        """
        var_dict = OrderedDict()

        print(f'len(names) is {len(names)}')
        print(f'len(variables) is {len(variables)}')
        print(f'len(gradients) is {len(gradients)}')
        for n, v, g in zip(names, variables, gradients):
            if 'moving' not in v.name and g != None:
                new_var = v - self._alpha * g
                print(f'{v.name} is updated')
                with open(args["output"] + "/log.txt", "a") as file:
                    file.write(f'{v.name} is updated\n\n')
            else:
                # batch norm statistics are copied as they are
                new_var = v
            var_dict[n] = new_var

        return var_dict

    
    def _build_var_dict(self,args):
        """
        Create a dictionary containing all the graph variables defined so far where names are the key and variable ops in the graph are the value
        """
        self._variables = self._net.get_all_variables()

        self._var_dict = OrderedDict()
        for v in self._variables:
            self._var_dict[v.name[:-2]]=v

    def _build_var_dict_fe(self,args):
        """
        Create a dictionary containing all the graph variables defined so far where names are the key and variable ops in the graph are the value
        """
        self._variables_fe = self._fe.get_all_variables()
        self._var_dict_fe = OrderedDict()
        for v in self._variables_fe:
            self._var_dict_fe[v.name[:-2]]=v

    def _build_var_dict_ph(self,args):
        """
        Create a dictionary containing all the graph variables defined so far where names are the key and variable ops in the graph are the value
        """
        self._variables_ph = self._ph.get_all_variables()
        self._var_dict_ph = OrderedDict()
        for v in self._variables_ph:
            self._var_dict_ph[v.name[:-2]]=v
    
    #========================ABSTRACT METHODs============================
    @abc.abstractmethod
    def _build_trainer(self, args):
        """
        Should build the training graph
        """
        pass

    @abc.abstractmethod
    def _perform_train_step(self):
        """
        Should do the magic with session example and whatever
        """
        pass
    
    @abc.abstractmethod
    def _perform_summary_step(self):
        """
        Should produce and return a summary string
        """
        pass
    
    @abc.abstractmethod
    def _setup_summaries(self):
        """
        Setup meta op to collect and visualize summaries
        """
        with tf.variable_scope('training'):
            self._summary_ops.append(tf.summary.image('prediction',preprocessing.colorize_img(self._predictions[-1],cmap='jet'),max_outputs=1))
            self._summary_ops.append(tf.summary.image('target-gt',preprocessing.colorize_img(self._target_summary,cmap='jet'),max_outputs=1))
            self._summary_ops.append(tf.summary.image('left',self._left_summary,max_outputs=1))
        
        self._merge_summaries()
        self._summary_ready = True

    @abc.abstractmethod
    def _validate_args(self, args):
        """
        Should validate the argument and add default values for the missing ones whic are not critical
        """
        # Check common args
        if 'model' not in args or not Nets.factory.checkExistance(args['model']):
            raise Exception('Unable to train without a valid model')
        if 'loss' not in args:
            raise Exception('Unable to train without a loss function ')
        if 'inputs' not in args:
            raise Exception("Unable to train without valid inputs")
        if 'left' not in args['inputs'] or 'right' not in args['inputs'] or 'target' not in args['inputs']:
            raise Exception("Missing left or right frame form inputs")
        if 'session' not in args:
            raise Exception('Unable to train without a session') 
        if "alpha" not in args:
            print("WARNING: alpha will be set to default 0.0001")
            args["alpha"]=0.0001
        if 'lr' not in args:
            print("WARNING: no lr specified, using default 0.0001")
            args['lr']=0.00001
    
        # save args value
        self._model = args['model']
        self._loss = args['loss']
        self._inputs = args['inputs']
        self._session = args['session']
        self._alpha = args['alpha']
        self._adaptation_steps = self._inputs['left'][0].get_shape()[0].value-1
        self._metaTaskPerBatch = max([self._inputs['left'].get_shape()[0].value, 1])

        with tf.variable_scope('utils'):
            self._global_step=tf.Variable(0,trainable=False,name='global_step')
            self._lr = tf.constant(args['lr'],name='lr')
        self._increment_global_step = tf.assign_add(self._global_step,1)
        
        self._optimizer = tf.train.AdamOptimizer(self._lr)
        self._optimizer_adapt = tf.train.AdamOptimizer(self._alpha)
    
    #========================PUBLIC METHOD===================================
    def perform_train_step(self,feed_dict=None):
        """
        Perform a training step and return the meta loss
        """
        self._check_for_ready()
        return self._perform_train_step(feed_dict)
    
    def perform_eval_step(self,feed_dict=None):
        """
        Compute the value of the loss function and returns it without updating the variables
        """
        self._check_for_ready()
        return self._perform_eval_step(feed_dict)
    
    def perform_summary_step(self,feed_dict=None):
        """
        Perform a summary step, produce a summary string and return it
        """
        self._check_for_ready()
        if not self._summary_ready:
            self._setup_summaries()
        return self._perform_summary_step(feed_dict)
    
    def get_prediction_ops(self):
        self._check_for_ready()
        return self._predictions
    
    def get_model(self):
        self._check_for_ready()
        return self._net
    
    def get_variables(self):
        self._check_for_ready()
        return self._all_variables

    def get_global_step(self):
        self._check_for_ready()
        return self._global_step
    
