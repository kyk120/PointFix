import tensorflow as tf
import numpy as np

from Nets import Stereo_net
from Nets import sharedLayers
#from Data_utils import preprocessing
from Nets.factory import register_net_to_factory

@register_net_to_factory()
class StandardPointHead(Stereo_net.StereoNet):
    def __init__(self, ph_config, args):
        super(StandardPointHead, self).__init__()

        self.ph_config = ph_config
        # fmt: off
        self.num_classes = ph_config['num_classes']
        self.fc_dim = ph_config['fc_dim']
        self.fine_grained_features_num = ph_config['fine_grained_features_num']
        self.num_fc = ph_config['num_fc']
        self.cls_agnostic_mask = ph_config['cls_agnostic_mask']
        self.coarse_pred_each_layer = ph_config['coarse_pred_each_layer']
        self.input_channels = ph_config['input_channels']
        # fmt: on

        self.fine_grained_features = args['fine_grained_features']
        self.post_features = args['post_features']
        self.refine = []

        if 'is_training' not in args:
            print('WARNING: flag for trainign not setted, using default False')
            args['is_training']=False
        if 'variable_collection' not in args:
            print('WARNING: no variable collection specified using the default one')
            args['variable_collection']=None

        # save args value
        self._variable_collection = args['variable_collection']
        self._isTraining=args['is_training']

    def build_networks(self):
        names = []
        with tf.variable_scope('point_rend/instance_norm'):

            names.append(f'point_rend/instance_norm')
            input_layer = self.fine_grained_features
            # self._add_to_layers(names[-1], tf.contrib.layers.instance_norm(input_layer))
            self._add_to_layers(names[-1], sharedLayers.batch_norm(input_layer, variable_collection=self._variable_collection))
            self.normed_fine_grained_features = self._get_layer_as_input(names[-1])

        with tf.variable_scope('point_rend/concat'):
            input_layer = tf.concat([self.normed_fine_grained_features, self.post_features], axis=-1)

        with tf.variable_scope('point_rend/point_rend'):

            names.append(f'point_rend/point_rend/fc1')
            self._add_to_layers(names[-1], sharedLayers.conv1d(input_layer, [1,self.input_channels, self.fc_dim], padding="SAME", name="fc1",variable_collection=self._variable_collection))

            names.append(f'point_rend/point_rend/fc2')
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv1d(input_layer, [1,self.fc_dim, self.fc_dim], padding="SAME", name="fc2",variable_collection=self._variable_collection))

            names.append(f'point_rend/point_rend/fc3')
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv1d(input_layer, [1,self.fc_dim, self.fc_dim], padding="SAME", name="fc3",variable_collection=self._variable_collection))

            names.append(f'point_rend/point_rend/fc4')
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv1d(input_layer, [1,self.fc_dim, 1], padding="SAME", name="fc4", activation=lambda x:x,variable_collection=self._variable_collection))

            self.refine = self._get_layer_as_input(names[-1])

    def get_refined(self):
        return self.refine

@register_net_to_factory()
class FeatureExtractor(Stereo_net.StereoNet):
    def __init__(self, fe_config, args, five_mode=False):
        super(FeatureExtractor, self).__init__()

        self.fe_config = fe_config

        self.input_channels = fe_config[0]
        self.hidden_channels1 = fe_config[1]

        if len(fe_config) == 3:
            self.output_channels = fe_config[2]
        else:
            self.hidden_channels2 = fe_config[2]
            self.output_channels = fe_config[3]

        self.five_mode = five_mode
        self.prediction = args['prediction']
        self.reprojection = args['reprojection']
        self.left = args['left']
        self.features = []

        if 'is_training' not in args:
            print('WARNING: flag for trainign not setted, using default False')
            args['is_training']=False
        if 'variable_collection' not in args:
            print('WARNING: no variable collection specified using the default one')
            args['variable_collection']=None

        # save args value
        self._variable_collection = args['variable_collection']
        self._isTraining=args['is_training']

    def build_networks(self):
        with tf.variable_scope('feature_extractor/concat'):
            input_layer = tf.concat([self.prediction, self.reprojection, self.left], axis=-1)

        with tf.variable_scope('feature_extractor'):
            names = []
            if not self.five_mode:
                names.append(f'feature_extractor/conv1')
                self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [3,3,self.input_channels, self.hidden_channels1], strides=2, padding="SAME", name="conv1",variable_collection=self._variable_collection))
                #x = sharedLayers.conv2d(input_layer, [3,3,input_channels, hidden_channels1], strides=2, padding="SAME", name="conv1")

                if len(self.fe_config) == 3:
                    names.append(f'feature_extractor/conv2')
                    input_layer = self._get_layer_as_input(names[-2])
                    self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [3, 3, self.hidden_channels1, self.output_channels], strides=1, padding="SAME", name="conv2",variable_collection=self._variable_collection))
                    #x = sharedLayers.conv2d(x, [3, 3, hidden_channels1, output_channels], strides=1, padding="SAME", name="conv2")
                else:
                    names.append(f'feature_extractor/conv2')
                    input_layer = self._get_layer_as_input(names[-2])
                    self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [3, 3, self.hidden_channels1, self.hidden_channels2], strides=1, padding="SAME", name="conv2",variable_collection=self._variable_collection))
                    #x = sharedLayers.conv2d(x, [3, 3, hidden_channels1, hidden_channels2], strides=1, padding="SAME", name="conv2")

                    names.append(f'feature_extractor/conv3')
                    input_layer = self._get_layer_as_input(names[-2])
                    self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [3, 3, self.hidden_channels2, self.output_channels], strides=1, padding="SAME", name="conv3",variable_collection=self._variable_collection))
                    #x = sharedLayers.conv2d(x, [3, 3, hidden_channels2, output_channels], strides=1, padding="SAME", name="conv3")
            else:
                names.append(f'feature_extractor/conv1')
                self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [5, 5, self.input_channels, self.hidden_channels1], strides=2, padding="SAME", name="conv1",variable_collection=self._variable_collection))
                #x = sharedLayers.conv2d(input_layer, [5, 5, input_channels, hidden_channels1], strides=2, padding="SAME", name="conv1")

                if len(self.fe_config) == 3:
                    names.append(f'feature_extractor/conv2')
                    input_layer = self._get_layer_as_input(names[-2])
                    self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [5, 5, self.hidden_channels1, self.output_channels], strides=1, padding="SAME", name="conv2",variable_collection=self._variable_collection))
                    #x = sharedLayers.conv2d(x, [5, 5, hidden_channels1, output_channels], strides=1, padding="SAME", name="conv2")
                else:
                    names.append(f'feature_extractor/conv2')
                    input_layer = self._get_layer_as_input(names[-2])
                    self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [5, 5, self.hidden_channels1, self.hidden_channels2], strides=1, padding="SAME", name="conv2",variable_collection=self._variable_collection))
                    #x = sharedLayers.conv2d(x, [5, 5, hidden_channels1, hidden_channels2], strides=1, padding="SAME", name="conv2")

                    names.append(f'feature_extractor/conv3')
                    input_layer = self._get_layer_as_input(names[-2])
                    self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [5, 5, self.hidden_channels2, self.output_channels], strides=1, padding="SAME", name="conv3",variable_collection=self._variable_collection))
                    #x = sharedLayers.conv2d(x, [5, 5, hidden_channels2, output_channels], strides=1, padding="SAME", name="conv3")

            self.features = self._get_layer_as_input(names[-1])
            #self.features = x

    def get_features(self):
        return self.features



def point_sample(imgs, coords):
    """
    Construct a new image by bilinear sampling from the input image.
    Points falling outside the source image boundary have value 0.
    Args:
        imgs: source image to be sampled from [batch, height_s, width_s, channels]
        coords: coordinates of source pixels to sample from [batch, height_t,width_t, 2]. height_t/width_t correspond to the dimensions of the outputimage (don't need to be the same as height_s/width_s). The two channels correspond to x and y coordinates respectively.
    Returns:
        A new sampled image [batch, height_t, width_t, channels]
    """

    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), [1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    with tf.name_scope('image_sampling'):
        coords_y, coords_x = tf.split(coords, [1, 1], axis=-1)
        inp_size = tf.shape(imgs)
        coord_size = tf.shape(coords)
        out_size = [coord_size[0],coord_size[1],inp_size[-1]]

        y_max = tf.cast(inp_size[1], 'float32')
        x_max = tf.cast(inp_size[2], 'float32')
        zero = tf.zeros([1], dtype='float32')

        #coords_x = tf.cast((coords_x + 1) * 0.5 * x_max, 'float32')
        #coords_y = tf.cast((coords_y + 1) * 0.5 * y_max, 'float32')

        coords_x = tf.cast(coords_x * x_max, 'float32')
        coords_y = tf.cast(coords_y * y_max, 'float32')

        x0 = tf.floor(coords_x + 0.5) - 0.5
        x1 = x0 + 1
        y0 = tf.floor(coords_y + 0.5) - 0.5
        y1 = y0 + 1

        wt_x0 = x1 - coords_x
        wt_x1 = coords_x - x0
        wt_y0 = y1 - coords_y
        wt_y1 = coords_y - y0

        x0_safe = tf.clip_by_value(tf.floor(x0), zero[0], x_max)
        y0_safe = tf.clip_by_value(tf.floor(y0), zero[0], y_max)
        x1_safe = tf.clip_by_value(tf.floor(x1), zero[0], x_max)
        y1_safe = tf.clip_by_value(tf.floor(y1), zero[0], y_max)

        ## indices in the flat image to sample from
        dim2 = tf.cast(inp_size[2], 'float32')
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
        #dim1 = tf.cast(inp_size[1], 'float32')
        base = tf.reshape(_repeat(tf.cast(tf.range(coord_size[0]), 'float32') * dim1, coord_size[1]), [out_size[0], out_size[1], 1])

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = x0_safe + base_y0
        idx01 = x0_safe + base_y1
        idx10 = x1_safe + base_y0
        idx11 = x1_safe + base_y1

        ## sample from imgs
        imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
        imgs_flat = tf.cast(imgs_flat, 'float32')
        im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
        im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
        im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
        im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        output = tf.add_n([
            w00 * im00, w01 * im01,
            w10 * im10, w11 * im11
        ])

        return output


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    #N = tf.shape(uncertainty_map)[0]
    #H = tf.shape(uncertainty_map)[1]
    #W = tf.shape(uncertainty_map)[2]
    N = uncertainty_map.get_shape()[0].value
    H = uncertainty_map.get_shape()[1].value
    W = uncertainty_map.get_shape()[2].value
    print(f'N is {N}')
    print(f'H is {H}')
    print(f'W is {W}')

    h_step = tf.cast(1/H, 'float32')
    w_step = tf.cast(1/W, 'float32')

    num_points = tf.minimum(H * W, tf.cast(num_points, 'int32'))
    result = tf.math.top_k(tf.reshape(uncertainty_map, [N, H*W]), k=num_points)
    point_indices = result.indices
    #point_coords = tf.zeros([N, num_points, 2], tf.int32)
    point_coords_abs_w = tf.cast(tf.mod(point_indices, W), 'int32')
    point_coords_abs_h = tf.cast(tf.math.floordiv(point_indices, W), 'int32')
    point_coords_abs = tf.concat([tf.expand_dims(point_coords_abs_h, axis=-1), tf.expand_dims(point_coords_abs_w, axis=-1)], axis=-1)

    point_coords_w = tf.divide(w_step, tf.cast(2.0, 'float32')) + tf.cast(tf.mod(point_indices, W), 'float32') * w_step
    point_coords_h = tf.divide(h_step, tf.cast(2.0, 'float32')) + tf.cast(tf.math.floordiv(point_indices, W), 'float32') * h_step
    #point_coords = tf.reshape(tf.concat([point_coords_w, point_coords_h], axis=0), [N, num_points, 2])
    point_coords = tf.concat([tf.expand_dims(point_coords_h, axis=-1), tf.expand_dims(point_coords_w, axis=-1)], axis=-1)
    #point_coords = [point_coords_w, point_coords_h]

    return point_coords_abs, point_coords  # , h_step, w_step









