import tensorflow as tf
import numpy as np

from MetaTrainer.factory import register_meta_trainer
from MetaTrainer import metaTrainer
from Data_utils import variables, preprocessing
from Losses import loss_factory
from Nets import sharedLayers

from Nets.auxiliary import StandardPointHead, FeatureExtractor, point_sample, get_uncertain_point_coords_on_grid

@register_meta_trainer()
class PointFix(metaTrainer.MetaTrainer):
    """
    Implementation of Learning to Adapt for Stereo
    """
    _learner_name = "PointFix"
    _valid_args = [
                      ("adaptationLoss", "loss for adaptation")
                  ] + metaTrainer.MetaTrainer._valid_args

    def __init__(self, **kwargs):
        super(PointFix, self).__init__(**kwargs)

        self._ready = True

    def _validate_args(self, args):
        super(PointFix, self)._validate_args(args)

        if "adaptationLoss" not in args:
            print("WARNING: no adaptation loss specified, defaulting to the same loss for training and adaptation")
            args['adaptationLoss'] = args['loss']

        self._adaptationLoss = args['adaptationLoss']
        self._adaptation_steps += 1

    def _setup_inner_loop_weights(self):
        self._w = [1.0] * self._adaptation_steps

    def _setup_gradient_accumulator(self):
        with tf.variable_scope('gradients_utils'):
            self._trainableVariables = [x for x in self._trainableVariables if x.trainable == True]
            self._gradAccum = [tf.Variable(tf.zeros_like(tv), trainable=False) for tv in self._trainableVariables]
            self._resetAccumOp = [tv.assign(tf.zeros_like(tv)) for tv in self._gradAccum]
            self._batchGrads = tf.gradients(self._lossOp, self._trainableVariables)
            self._accumGradOps = [accum.assign_add(grad) for accum, grad in zip(self._gradAccum, self._batchGrads)]
            self._gradients_to_be_applied_ops = [grad / self._metaTaskPerBatch for grad in self._gradAccum]
            self._train_op = self._optimizer.apply_gradients([(grad, var) for grad, var in zip(self._gradients_to_be_applied_ops, self._trainableVariables)])

    def _build_adaptation_loss(self, gt_logits, refinement, abs_err):
        total_loss = tf.reduce_mean(tf.abs(gt_logits - refinement))
        return total_loss

    def _first_inner_loop_step(self, args):

        self._trainableVariables_base = self._net.get_trainable_variables()
        self._trainableVariables_fe = self._fe.get_trainable_variables()
        self._trainableVariables_ph = self._ph.get_trainable_variables()
        self._trainableVariables = self._trainableVariables_base + self._trainableVariables_fe + self._trainableVariables_ph

        self._all_variables_base = self._net.get_all_variables()
        self._all_variables_fe = self._fe.get_all_variables()
        self._all_variables_ph = self._ph.get_all_variables()
        self._all_variables = self._all_variables_base + self._all_variables_fe + self._all_variables_ph

        self._variableState = variables.VariableState(self._session, self._all_variables)

        self._predictions = self._net.get_disparities()

        self._build_var_dict(args)
        self._build_var_dict_fe(args)
        self._build_var_dict_ph(args)

    def network_setting(self, inputs, new_model_var, fe_var, ph_var):

        net = self._build_forward(inputs['left'], inputs['right'], new_model_var)
        current_disparity = net.get_disparities()
        full_res_disp = current_disparity[-1]
        full_res_shape = inputs['left'].get_shape().as_list()
        full_res_shape[-1] = 1
        full_res_disp.set_shape(full_res_shape)

        self._full_res_disp = full_res_disp

        if net._name == 'Dispnet':
            corr = tf.concat([net._layers['corr'], net._layers['conv2a']], axis=3)
        elif net._name == 'MADNet':
            corr = net._layers['corr_2']
        else:
            print(f"Don't know what to do with corr!")

        self._net = net

        gt_input = tf.where(tf.is_finite(inputs['target']), inputs['target'], tf.zeros_like(inputs['target']))

        valid_map = tf.cast(tf.logical_not(tf.equal(gt_input, 0)), tf.float32)
        gt_input_disp = gt_input - 1 + valid_map

        targets_disp = tf.clip_by_value(self._dataset_param * tf.math.reciprocal(gt_input_disp), -1, 1000)
        targets_disp = targets_disp * valid_map

        fe_config = [5, 32, 64, 128]
        fe_args = {
            'prediction': full_res_disp / 128,  # MAX_DISP,
            'reprojection': targets_disp / 128,
            'left': inputs['left'] / 255,
            'variable_collection': fe_var
        }

        fe = FeatureExtractor(fe_config, fe_args, five_mode=False)
        fe.build_networks()
        self._fe = fe
        print(f'fe is {fe}')

        post_feature_map = fe.get_features()
        print(f'post_feature_map is {post_feature_map}')

        # compute error against gt
        abs_err = tf.abs(full_res_disp - targets_disp)
        # valid_map = tf.cast(tf.logical_not(tf.equal(gt_input, 0)), tf.float32)
        filtered_error = abs_err * valid_map

        abs_err = tf.reduce_sum(filtered_error) / tf.reduce_sum(valid_map)
        self.abs_err = abs_err
        error_pixels = tf.greater(filtered_error, 3)
        bad_pixel_abs = tf.cast(error_pixels, tf.float32)
        bad_pixel_perc = tf.reduce_sum(bad_pixel_abs) / tf.reduce_sum(valid_map)

        bad_pixels_num = tf.reduce_sum(bad_pixel_abs)

        point_indices, point_coords = get_uncertain_point_coords_on_grid(bad_pixel_abs, num_points=bad_pixels_num)
        print(f'point_coords.shape is {point_coords.shape}')
        print(f'point_indices.shape is {point_indices.shape}')

        fine_grained_features = point_sample(corr, point_coords)
        print(f'fine_grained_features.shape is {fine_grained_features.shape}')

        coarse_features = point_sample(full_res_disp / 128, point_coords)  # MAX_DISP
        print(f'coarse_features.shape is {coarse_features.shape}')

        post_features = point_sample(post_feature_map, point_coords)
        print(f'post_features.shape is {post_features.shape}')

        gt_logits = point_sample(targets_disp, point_coords)
        print(f'gt_logits.shape is {gt_logits.shape}')
        self._gt_logits = gt_logits

        print(f'fine_grained_features.shape[-1] is {fine_grained_features.shape[-1]}')
        print(f'post_features.shape[-1] is {post_features.shape[-1]}')

        ph_config = {
            'num_classes': 1,
            'fc_dim': fine_grained_features.shape[-1] + post_features.shape[-1],  # - 1,
            'fine_grained_features_num': fine_grained_features.shape[-1],
            'num_fc': 3,
            'cls_agnostic_mask': True,
            'coarse_pred_each_layer': False,
            'input_channels': fine_grained_features.shape[-1] + post_features.shape[-1]  # - 1
        }

        ph_args = {
            'fine_grained_features': fine_grained_features,
            'post_features': post_features,
            'variable_collection': ph_var
        }

        ph = StandardPointHead(ph_config, ph_args)
        ph.build_networks()
        self._ph = ph
        print(f'ph is {ph}')

        refine = ph.get_refined()
        print(f'refine is {refine}')

        refinement = (refine + coarse_features) * 128  # MAX_DISP
        self._refinement = refinement

        print(f'full_res_disp b is {full_res_disp}')
        print(f'point_indices b is {point_indices}')
        print(f'refinement b is {refinement}')

        refined = tf.tensor_scatter_update(tf.squeeze(full_res_disp, axis=0), tf.squeeze(point_indices, axis=0), tf.squeeze(refinement, axis=0))
        refined = tf.expand_dims(refined, axis=0)
        # refined = full_res_disp + refine
        print(f'refined is {refined}')


    def _build_updater(self, args):
        input_shape = self._inputs['left'].get_shape().as_list()
        input_shape = [input_shape[0]*input_shape[1], input_shape[2], input_shape[3], input_shape[4]]
        target_shape = input_shape[:3] + [1]

        print(f'input_shape in update is {input_shape}')

        self._left_input_placeholder_updater = tf.placeholder(dtype=tf.float32, shape=input_shape)
        self._right_input_placeholder_updater = tf.placeholder(dtype=tf.float32, shape=input_shape)
        self._target_placeholder_updater = tf.placeholder(dtype=tf.float32, shape=target_shape)

        print(f'self._left_input_placeholder_updater is {self._left_input_placeholder_updater}')
        print(f'self._right_input_placeholder_updater is {self._right_input_placeholder_updater}')
        print(f'self._target_placeholder_updater is {self._target_placeholder_updater}')

        inputs = {}
        inputs['left'] = self._left_input_placeholder_updater
        inputs['right'] = self._right_input_placeholder_updater
        inputs['target'] = self._target_placeholder_updater

        #"""
        net = self._build_forward(inputs['left'], inputs['right'], self._var_dict)
        self._net_update = net
        #"""

        current_disparity = net.get_disparities()
        full_res_disp = current_disparity[-1]
        full_res_shape = inputs['left'].get_shape().as_list()
        full_res_shape[-1] = 1
        full_res_disp.set_shape(target_shape)
        self._full_res_disp_update = full_res_disp

        if net._name == 'Dispnet':
            corr = tf.concat([net._layers['corr'], net._layers['conv2a']], axis=3)
        elif net._name == 'MADNet':
            corr = net._layers['corr_2']
        else:
            print(f"Don't know what to do with corr!")

        gt_input = tf.where(tf.is_finite(inputs['target']), inputs['target'], tf.zeros_like(inputs['target']))

        valid_map = tf.cast(tf.logical_not(tf.equal(gt_input, 0)), tf.float32)
        gt_input_disp = gt_input - 1 + valid_map

        targets_disp = tf.clip_by_value(self._dataset_param * tf.math.reciprocal(gt_input_disp), -1, 1000)
        targets_disp = targets_disp * valid_map

        fe_config = [5, 32, 64, 128]
        fe_args = {
            'prediction': full_res_disp / 128,  # MAX_DISP,
            'reprojection': targets_disp / 128,
            'left': inputs['left'] / 255,
            'variable_collection': self._var_dict_fe
        }
        fe = FeatureExtractor(fe_config, fe_args, five_mode=False)
        fe.build_networks()

        print(f'fe update is {fe}')

        post_feature_map = fe.get_features()

        # compute error against gt
        print(f'full_res_disp is {full_res_disp}')
        print(f'targets_disp is {targets_disp}')
        abs_err = tf.abs(full_res_disp - targets_disp)
        # valid_map = tf.cast(tf.logical_not(tf.equal(gt_input, 0)), tf.float32)
        filtered_error = abs_err * valid_map

        abs_err = tf.reduce_sum(filtered_error) / tf.reduce_sum(valid_map)
        self.abs_err_update = abs_err
        error_pixels = tf.greater(filtered_error, 3)
        bad_pixel_abs = tf.cast(error_pixels, tf.float32)
        bad_pixel_perc = tf.reduce_sum(bad_pixel_abs) / tf.reduce_sum(valid_map)

        bad_pixels_num = tf.reduce_sum(bad_pixel_abs)

        point_indices, point_coords = get_uncertain_point_coords_on_grid(bad_pixel_abs, num_points=bad_pixels_num)
        print(f'point_coords.shape is {point_coords.shape}')
        print(f'point_indices.shape is {point_indices.shape}')

        fine_grained_features = point_sample(corr, point_coords)
        print(f'fine_grained_features.shape is {fine_grained_features.shape}')

        coarse_features = point_sample(full_res_disp / 128, point_coords)  # MAX_DISP
        print(f'coarse_features.shape is {coarse_features.shape}')

        post_features = point_sample(post_feature_map, point_coords)
        print(f'post_features.shape is {post_features.shape}')

        gt_logits = point_sample(targets_disp, point_coords)
        print(f'gt_logits.shape is {gt_logits.shape}')
        self._gt_logits_update = gt_logits

        print(f'fine_grained_features.shape[-1] is {fine_grained_features.shape[-1]}')
        print(f'post_features.shape[-1] is {post_features.shape[-1]}')

        ph_config = {
            'num_classes': 1,
            'fc_dim': fine_grained_features.shape[-1] + post_features.shape[-1],  # - 1,
            'fine_grained_features_num': fine_grained_features.shape[-1],
            'num_fc': 3,
            'cls_agnostic_mask': True,
            'coarse_pred_each_layer': False,
            'input_channels': fine_grained_features.shape[-1] + post_features.shape[-1]  # - 1
        }

        ph_args = {
            'fine_grained_features': fine_grained_features,
            'post_features': post_features,
            'variable_collection': self._var_dict_ph
        }

        ph = StandardPointHead(ph_config, ph_args)
        ph.build_networks()
        self._ph_update = ph

        print(f'ph update is {ph}')

        refine = ph.get_refined()
        print(f'refine is {refine}')

        refinement = (refine + coarse_features) * 128  # MAX_DISP
        self._refinement_update = refinement

        print(f'full_res_disp is {full_res_disp}')
        print(f'point_indices is {point_indices}')
        print(f'refinement is {refinement}')

        iter = input_shape[0]
        print(f'iter is {iter}')

        refined = []
        for i in range(iter):
            print(f'full_res_disp[{i}] is {full_res_disp[i]}')
            print(f'point_indices[{i}] is {point_indices[i]}')
            print(f'refinement[{i}] is {refinement[i]}')
            refined_n = tf.tensor_scatter_update(full_res_disp[i], point_indices[i], refinement[i])
            refined_n = tf.expand_dims(refined_n, axis=0)
            refined.append(refined_n)
        print(f'refined is {refined}')
        refined = tf.concat(refined, 0)

        print(f'refined is {refined}')

        self.point_loss_update = self._build_adaptation_loss(self._gt_logits_update, self._refinement_update, self.abs_err_update)
        self._update_op = self._optimizer_adapt.minimize(self.point_loss_update)

    def _build_trainer(self, args):
        # build model taking placeholder as input
        input_shape = self._inputs['left'].get_shape().as_list()
        print(f'input_shape is {input_shape}')

        self._left_input_placeholder = tf.placeholder(dtype=tf.float32, shape=input_shape[1:])
        self._right_input_placeholder = tf.placeholder(dtype=tf.float32, shape=input_shape[1:])
        self._target_placeholder = tf.placeholder(dtype=tf.float32, shape=input_shape[1:4] + [1])

        new_model_var = None
        fe_var = None
        ph_var = None

        loss_collection = []
        self._adaptation_loss = []
        self._loss_collection = []
        self._abs_err = []
        self.disp_map = []
        print(f'self._adaptation_steps + 1 is {self._adaptation_steps + 1}')
        for i in range(self._adaptation_steps + 1):
            # forward pass
            inputs = {}

            inputs['left'] = self._left_input_placeholder
            inputs['right'] = self._right_input_placeholder
            inputs['target'] = self._target_placeholder

            self.network_setting(inputs, new_model_var, fe_var, ph_var)

            if i != self._adaptation_steps:
                # compute loss and gradients
                adapt_loss = self._build_adaptation_loss(self._gt_logits, self._refinement, self.abs_err)
                self._adaptation_loss.append(adapt_loss)

            if i == 0:
                # Create variable state to handle variable updates and reset
                self._first_inner_loop_step(args)
                new_model_var = self._var_dict
                fe_var = self._var_dict_fe
                ph_var = self._var_dict_ph

            else:
                current_loss = self._loss([self._full_res_disp], inputs, dataset_param=self._dataset_param)
                loss_collection.append(current_loss)
                self.disp_map.append(self._full_res_disp)

            if i != self._adaptation_steps:
                # build updated variables

                print(f'\n{i}th new_model_var is {new_model_var}')
                print(f'{i}th type(new_model_var) is {type(new_model_var)}\n')
                print(f'\n{i}th fe_var is {fe_var}')
                print(f'\n{i}th ph_var is {ph_var}')

                with open(args["output"] + "/log.txt", "a") as file:
                    file.write(f'\n{i}th new_model_var is {new_model_var}')
                    file.write(f'\n{i}th type(new_model_var) is {type(new_model_var)}\n')

                gradients_base = tf.gradients(adapt_loss, list(new_model_var.values()))
                gradients_fe = tf.gradients(adapt_loss, list(fe_var.values()))
                gradients_ph = tf.gradients(adapt_loss, list(ph_var.values()))

                print(f'\n')
                print(f'gradients is {gradients_base}')
                print(f'gradients_fe is {gradients_fe}')
                print(f'gradients_ph is {gradients_ph}')
                with open(args["output"] + "/log.txt", "a") as file:
                    file.write(f'\ngradients_base is {gradients_base}\n')
                    file.write(f'\ngradients_fe is {gradients_fe}\n')
                    file.write(f'\ngradients_ph is {gradients_ph}\n')

                new_model_var = self._build_updated_variables(list(self._var_dict.keys()), list(new_model_var.values()), gradients_base, args)
                fe_var = self._build_updated_variables(list(self._var_dict_fe.keys()), list(fe_var.values()), gradients_fe, args)
                ph_var = self._build_updated_variables(list(self._var_dict_ph.keys()), list(ph_var.values()), gradients_ph, args)

        self._new_model_var = new_model_var
        self._fe_var = fe_var
        self._ph_var = ph_var

        self._loss_collection = loss_collection
        self._setup_inner_loop_weights()
        assert (len(self._w) == len(loss_collection))
        self._lossOp = tf.reduce_sum([w * l for w, l in zip(self._w, loss_collection)])

        # create accumulator for gradients to get batch gradients
        self._setup_gradient_accumulator()

    def _perform_train_step(self, feed_dict=None):
        # read all the input data and reset gradients accumulator
        left_images, right_images, target_images, _ = self._session.run([self._inputs['left'], self._inputs['right'], self._inputs['target'], self._resetAccumOp], feed_dict=feed_dict)

        # read variable
        var_initial_state = self._variableState.export_variables()

        # for all tasks
        loss = 0
        loss_collection = []
        adaptation_loss = []
        maps = []
        #print(f'self._metaTaskPerBatch is {self._metaTaskPerBatch}')
        for task_id in range(self._metaTaskPerBatch):
            # perform adaptation and evaluation for a single task/video sequence

            fd = {
                self._left_input_placeholder: left_images[task_id, :, :, :, :],
                self._right_input_placeholder: right_images[task_id, :, :, :, :],
                self._target_placeholder: target_images[task_id, :, :, :, :],
            }
            if feed_dict is not None:
                fd.update(feed_dict)
            _, ll, collection, adaptation, map = self._session.run([self._accumGradOps, self._lossOp, self._loss_collection, self._adaptation_loss, self.disp_map], feed_dict=fd)
            loss += ll
            loss_collection.append(collection)
            adaptation_loss.append(adaptation)
            maps.append(map)

            # reset vars
            self._variableState.import_variables(var_initial_state)

        # apply accumulated grads to meta learn
        _, self._step_eval = self._session.run([self._train_op, self._increment_global_step], feed_dict=feed_dict)
        input_shape = left_images.shape
        input_shape = [input_shape[0] * input_shape[1], input_shape[2], input_shape[3], input_shape[4]]
        target_shape = input_shape[:3] + [1]

        left_images_update = left_images.reshape(input_shape)
        right_images_update = right_images.reshape(input_shape)
        target_images_update = target_images.reshape(target_shape)

        fd_update = {
            self._left_input_placeholder_updater: left_images_update,
            self._right_input_placeholder_updater: right_images_update,
            self._target_placeholder_updater: target_images_update,
        }
        _, point_loss = self._session.run([self._update_op, self.point_loss_update], feed_dict=fd_update)

        return self._step_eval, loss / self._metaTaskPerBatch, loss_collection, adaptation_loss, maps, left_images, right_images, target_images

    def _setup_summaries(self):
        with tf.variable_scope('base_model_output'):
            self._summary_ops.append(tf.summary.image('left', self._left_input_placeholder, max_outputs=1))
            self._summary_ops.append(
                tf.summary.image('target_gt', preprocessing.colorize_img(self._target_placeholder, cmap='jet'),
                                 max_outputs=1))
            self._summary_ops.append(
                tf.summary.image('prediction', preprocessing.colorize_img(self._predictions[-1], cmap='jet'),
                                 max_outputs=1))
        self._merge_summaries()
        self._summary_ready = True

    def _perform_summary_step(self, feed_dict=None):
        # read one batch of data
        left_images, right_images, target_images, _ = self._session.run(
            [self._inputs['left'], self._inputs['right'], self._inputs['target'], self._resetAccumOp],
            feed_dict=feed_dict)

        # for first task
        task_id = 0

        # perform meta task
        fd = {
            self._left_input_placeholder: left_images[task_id, :, :, :, :],
            self._right_input_placeholder: right_images[task_id, :, :, :, :],
            self._target_placeholder: target_images[task_id, :, :, :, :]
        }

        if feed_dict is not None:
            fd.update(feed_dict)
        summaries, step = self._session.run([self._summary_op, self._increment_global_step], feed_dict=fd)
        return step, summaries