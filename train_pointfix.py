import tensorflow as tf
import argparse
import os
import sys
import time
import datetime

import cv2

import Nets
from Data_utils import data_reader, weights_utils
import MetaTrainer
from Losses import loss_factory


def get_loss(name, unSupervised, masked=True, dataset_param=None):
    if unSupervised:
        return loss_factory.get_reprojection_loss(name, False, False)
    else:
        return loss_factory.get_supervised_loss(name, False, False, mask=masked, dataset_param=dataset_param)


def main(args):

    gpu_options = tf.GPUOptions(allow_growth=False, visible_device_list=args.gpu_num)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # build input producer
        with tf.variable_scope('train_reader'):
            print('Building input pipeline')
            data_set = data_reader.metaDataset(
                args.dataset,
                batch_size=args.batchSize,
                sequence_length=args.adaptationSteps,
                resize_shape=args.resizeShape,
                crop_shape=args.cropShape,
                augment=args.augment,
                original_shape=args.originalShape)

            left_train_batch, right_train_batch, gt_train_batch, dataset_param_batch = data_set.get_batch()

        # Build meta trainer
        with tf.variable_scope('train_model') as scope:
            print('Building meta trainer')
            # build params
            input_meta_train = {}
            input_meta_train['left'] = left_train_batch
            input_meta_train['right'] = right_train_batch
            input_meta_train['target'] = gt_train_batch

            t_args = {}
            t_args['inputs'] = input_meta_train
            t_args['model'] = args.modelName

            if args.dataset_param == 'KITTI':
                dataset_param = 380.0
            else:
                dataset_param = 166.0

            t_args['dataset_param'] = dataset_param

            masked = args.maskedGT
            t_args['loss'] = get_loss(args.loss, args.unSupervised, masked=masked, dataset_param=dataset_param)
            t_args['adaptationLoss'] = get_loss(args.adaptationLoss, args.unSupervisedMeta, masked=masked)

            t_args['lr'] = args.lr
            t_args['alpha'] = args.alpha
            t_args['session'] = sess
            t_args['weighting_net'] = args.weighting_net
            t_args['output'] = args.output

            # build meta trainer
            meta_learner = MetaTrainer.factory.get_meta_learner(args.metaAlgorithm, t_args)

            # placeholder to log meta_loss
            meta_loss = tf.placeholder(dtype=tf.float32)
            tf.summary.scalar('meta_loss', meta_loss)

        # build meta op to save progress
        print('Building periodical saver')
        # add summaries
        summary_op = tf.summary.merge_all()
        logger = tf.summary.FileWriter(args.output)

        # create saver
        train_vars = meta_learner.get_variables()
        print(f'train_vars to save is {train_vars}')
        main_saver = tf.train.Saver(var_list=train_vars, max_to_keep=0)

        print('Everything ready, start training')

        # init stuff
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        if args.retrain:
            ignore_list = ['train_model/']
        else:
            ignore_list = ['model/']

        # restore disparity inference weights
        restored, step = weights_utils.check_for_weights_or_restore_them(args.output, sess,
                                                                         initial_weights=args.weights,
                                                                         prefix=args.prefix,
                                                                         ignore_list=ignore_list)  # ignore_list=['model/'] # ignore_list=['train_model/']
        print('Disparity Net Restored?: {}'.format(restored))

        # restore step
        global_step = meta_learner.get_global_step()
        sess.run(global_step.assign(step))

        # try:

        start_time = time.time()
        estimated_step = args.numStep
        step_eval = step
        loss_acc = 0

        start_step = 0
        total_step = 0
        if args.retrain:
            start_step = args.retrain

        step_count = start_step

        with open(args.output + "/log.txt", "a") as file:
            file.write(f"start training at {datetime.datetime.now()}\n\n")

        while step_count < estimated_step:  # step_eval
            # train step
            if args.metaAlgorithm == 'FineTuner':
                step_eval, loss_eval = meta_learner.perform_train_step()
            else:
                step_eval, loss_eval, l_c, a_l, maps, left_images, right_images, target_images = meta_learner.perform_train_step()
            loss_acc += loss_eval

            step_count += 1
            total_step += 1
            if step_count % 10 == 0:
                print(f'step count: {step_count} step Total step: {total_step} step!')
                print(f'loss_eval: {loss_eval}')

                with open(args.output + "/log.txt", "a") as file:
                    file.write(f'step count: {step_count} step Total step: {total_step} step!\n')
                    file.write(f'loss_eval: {loss_eval}\n\n')

            if step_count % 50 == 0:

                end_time = time.time()
                elapsed_time = end_time - start_time
                missing_time = ((estimated_step - step_count) // 50) * elapsed_time
                print(f'Step: {step_eval}, step count: {step_count} Total step: {total_step} Loss: {loss_acc / 50}, f/b time x 50 iteration: {elapsed_time}s, missing time: {datetime.timedelta(seconds=missing_time)}')
                if args.metaAlgorithm != 'FineTuner':
                    print('loss_collection: {}'.format(l_c))
                    print('adaptation_loss: {}'.format(a_l))
                print('now: {}'.format(datetime.datetime.now()))

                with open(args.output + "/log.txt", "a") as file:
                    file.write(f'Step: {step_eval}, step count: {step_count} Total step: {total_step}, Loss: {loss_acc / 50}, f/b time x 50 iteration: {elapsed_time}s, missing time: {datetime.timedelta(seconds=missing_time)}\n')
                    if args.metaAlgorithm != 'FineTuner':
                        file.write(f'loss_collection: {l_c}\n')
                        file.write(f'adaptation_loss: {a_l}\n')
                    file.write(f'now: {datetime.datetime.now()}\n\n')

                fd = {meta_loss: loss_acc / 50}
                loss_acc = 0

                if step_count % 5000 == 0:
                    if not os.path.exists(os.path.join(args.output, f'main_saver')):
                        os.makedirs(os.path.join(args.output, f'main_saver'), exist_ok=True)
                    main_saver.save(sess, os.path.join(args.output, f'main_saver/weights.ckpt'), global_step=step_count)

                # Scalar summaries
                summary_string = sess.run(summary_op, feed_dict=fd)
                logger.add_summary(summary_string, global_step=step_eval)

                # image summaries
                step_eval, internal_summary = meta_learner.perform_summary_step()
                logger.add_summary(internal_summary, global_step=step_eval)
                start_time = time.time()

                step_count += 1
                total_step += 1

        print(f'{step_count} step parameters saved with loss_acc/50: {loss_acc / 50}  loss_eval: {loss_eval}')
        with open(args.output + "/log.txt", "a") as file:
            file.write(
                f'{step_count} step parameters saved with loss_acc/50: {loss_acc / 50}  loss_eval: {loss_eval}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to train a Stereo net with learning to adapt for stereo")
    parser.add_argument("--dataset", required=True, type=str, help='paths to the dataset list files')
    parser.add_argument("-o", "--output", type=str,
                        help='path to the output folder where results and logs will be saved', required=True)
    parser.add_argument("-b", "--batchSize", default=4, type=int, help='batch size')
    parser.add_argument("-w", "--weights", help="preinitialization weights", metavar="FILE", default=None)
    parser.add_argument("-n", "--numStep", type=int, default=100000, help='number of training steps')
    parser.add_argument("--modelName", help="name of the stereo model to train", default="Dispnet",
                        choices=Nets.factory.getAvailableNets())
    parser.add_argument("--adaptationSteps", help="number of frames for each meta task", default=1, type=int)
    parser.add_argument("--lr", help="value for learning rate", default=0.0001, type=float)
    parser.add_argument("--prefix", help='prefix to be added to the pretrained weights before restoring them',
                        default='train_model/')
    parser.add_argument("--resizeShape",
                        help="Two int for the resize shape [height,width], leave to default for no resize", nargs='+',
                        type=int, default=[None, None])
    parser.add_argument("--cropShape", help='two int for the crop shape [height,width], leave to default for no crop',
                        nargs='+', default=[None, None], type=int)
    parser.add_argument("--augment", help='flag to enable augmentation of training inputs', action='store_true')
    parser.add_argument("--loss", help="type of loss function to be used for evaluation (L_s)",
                        choices=loss_factory.SUPERVISED_LOSS.keys(), default='mean_L1')
    parser.add_argument("--adaptationLoss", help="type of loss function to be used for adaptation (L_u)",
                        choices=loss_factory.SUPERVISED_LOSS.keys(), default='mean_l1')
    parser.add_argument("--unSupervised",
                        help="Flag to use left right reprojection when computing L_s instead of using the gt",
                        action='store_true')
    parser.add_argument("--unSupervisedMeta",
                        help="Flag to use left right reprojection when computing L_u instead of using the gt",
                        action='store_true')
    parser.add_argument("--alpha", help="learning rate of the inner optimization loop for meta training", type=float,
                        default=0.0001)
    parser.add_argument("--metaAlgorithm", help="name of the meta algorithm to use",
                        choices=MetaTrainer.factory.get_available_meta_learner(), required=True)
    parser.add_argument("--maskedGT",
                        help="Flag to enable the use of ground truth data where invalid pixels are marked as zeros",
                        action='store_true')
    parser.add_argument("--dataset_param", help="dataset parameters to convert depth and disparity",
                        choices=['Synthia', 'KITTI'], default='Synthia')
    parser.add_argument("--retrain", help="retrained parameter final step", type=int)
    parser.add_argument("--weighting_net", help="apply weighting_network", action='store_true')
    parser.add_argument("--gpu_num", help="gpu index to run", default="0")
    parser.add_argument("--originalShape", help="Flag to maintain original shape of input image", action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'params.sh'), 'w+') as out:
        sys.argv[0] = os.path.join(os.getcwd(), sys.argv[0])
        out.write('#!/bin/bash\n')
        out.write('python3 ')
        out.write(' '.join(sys.argv))
        out.write('\n')

    if not os.path.exists(os.path.join(args.output, 'codes')):
        os.makedirs(os.path.join(args.output, 'codes'))

    import shutil

    shutil.copytree('./Data_utils', os.path.join(args.output, 'codes/Data_utils'))
    shutil.copytree('./Losses', os.path.join(args.output, 'codes/Losses'))
    shutil.copytree('./MetaTrainer', os.path.join(args.output, 'codes/MetaTrainer'))
    shutil.copytree('./Nets', os.path.join(args.output, 'codes/Nets'))
    shutil.copy('train_pointfix.py', os.path.join(args.output, 'codes/train_pointfix.py'))
    shutil.copy('test_online_adaptation.py', os.path.join(args.output, 'codes/test_online_adaptation.py'))

    main(args)



#############################################


