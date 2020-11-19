import os
import pickle
import sys

import numpy as np
import scipy.stats
from sklearn.metrics import f1_score

from vmaf.tools.decorator import override

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    tf = None

from vmaf.core.h5py_mixin import H5pyMixin
from vmaf.core.train_test_model import RawVideoTrainTestModelMixin, TrainTestModel, \
    ClassifierMixin
from vmaf.tools.sigproc import as_one_hot, create_hp_yuv_4channel, dstack_y_u_v
from vmaf.tools.misc import get_dir_without_last_slash

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class NeuralNetTrainTestModel(RawVideoTrainTestModelMixin,
                              TrainTestModel,
                              # order affects whose _assert_dimension
                              # gets called
                              H5pyMixin):

    DEFAULT_N_EPOCHS = 30
    DEFAULT_LEARNING_RATE = 1e-3
    DEFAULT_PATCH_WIDTH = 50
    DEFAULT_PATCH_HEIGHT = 50
    DEFAULT_PATCHES_PER_FRAME = 1
    DEFAULT_BATCH_SIZE = 20
    DEFAULT_SEED = None

    @property
    def n_epochs(self):
        if self.param_dict is not None and 'n_epochs' in self.param_dict:
            return self.param_dict['n_epochs']
        else:
            return self.DEFAULT_N_EPOCHS

    @property
    def learning_rate(self):
        if self.param_dict is not None and 'learning_rate' in self.param_dict:
            return self.param_dict['learning_rate']
        else:
            return self.DEFAULT_LEARNING_RATE

    @property
    def patch_width(self):
        if self.param_dict is not None and 'patch_width' in self.param_dict:
            return self.param_dict['patch_width']
        else:
            return self.DEFAULT_PATCH_WIDTH

    @property
    def patch_height(self):
        if self.param_dict is not None and 'patch_height' in self.param_dict:
            return self.param_dict['patch_height']
        else:
            return self.DEFAULT_PATCH_HEIGHT

    @property
    def patches_per_frame(self):
        if self.param_dict is not None and \
           'patches_per_frame' in self.param_dict:
            return self.param_dict['patches_per_frame']
        else:
            return self.DEFAULT_PATCHES_PER_FRAME

    @property
    def batch_size(self):
        if self.param_dict is not None and 'batch_size' in self.param_dict:
            return self.param_dict['batch_size']
        else:
            return self.DEFAULT_BATCH_SIZE

    @property
    def seed(self):
        if self.param_dict is not None and 'seed' in self.param_dict:
            return self.param_dict['seed']
        else:
            return self.DEFAULT_SEED

    @property
    def checkpoints_dir(self):
        # if None, won't output checkpoints
        if self.optional_dict2 is not None and \
           'checkpoints_dir' in self.optional_dict2:
            return self.optional_dict2['checkpoints_dir']
        else:
            return None

    def _assert_args(self):
        super(NeuralNetTrainTestModel, self)._assert_args()
        self.assert_h5py_file() # assert h5py_file in self.optional_dict2

    @staticmethod
    def _assert_xs(xs):
        # for now, force input xys or xs having 'dis_y', 'dis_u' and 'dis_v'
        assert 'dis_y' in xs
        assert 'dis_u' in xs
        assert 'dis_v' in xs

    @override(TrainTestModel)
    def train(self, xys):
        self._assert_xs(xys)

        self.model_type = self.TYPE

        assert 'label' in xys
        assert 'content_id' in xys

        # this makes sure the order of features are normalized, and each
        # dimension of xys_2d is consistent with feature_names
        feature_names = sorted(xys.keys())
        feature_names.remove('label')
        feature_names.remove('content_id')
        self.feature_names = feature_names

        self.norm_type = 'none' # no conventional data normalization

        patches_cache, labels_cache = self._populate_patches_and_labels(
            feature_names, xys)

        model = self._train(patches_cache, labels_cache)
        self.model = model

    @override(TrainTestModel)
    def predict(self, xs):

        self._assert_xs(xs)
        self._assert_trained()
        for name in self.feature_names:
            assert name in xs

        feature_names = self.feature_names

        # loop through xs
        len_xs = len(xs[feature_names[0]])
        ys_label_pred = []
        for i in range(len_xs):

            sys.stdout.write("Evaluating test data: %d / %d\r" % (i, len_xs))
            sys.stdout.flush()

            # create single x
            x = {}
            for feature_name in feature_names:
                x[feature_name] = [xs[feature_name][i]]

            # extract patches
            patches_cache = self._populate_patches(feature_names, x)

            # predict
            y_label_pred = self._predict(self.model, patches_cache)
            ys_label_pred.append(y_label_pred)

        return {'ys_label_pred': ys_label_pred}

    def _create_patch_and_label_dataset(self, total_frames, overwrite=True):

        patches_dims = (total_frames * self.patches_per_frame,
                        self.patch_height, self.patch_width, self.n_channels)
        if overwrite:
            try:
                del self.h5py_file['patches']
            except KeyError:
                pass
        patches_cache = self.h5py_file.create_dataset('patches', patches_dims, dtype='float')
        patches_cache.dims[0].label = 'batch'
        patches_cache.dims[1].label = 'height'
        patches_cache.dims[2].label = 'width'
        patches_cache.dims[3].label = 'channel'

        labels_dims = (total_frames * self.patches_per_frame, )
        if overwrite:
            try:
                del self.h5py_file['labels']
            except KeyError:
                pass
        labels_cache = self.h5py_file.create_dataset('labels', labels_dims, dtype='uint8')

        return patches_cache, labels_cache

    def _populate_patches_and_labels(self, xkeys, xys, mode='train'):

        np.random.seed(self.seed)

        # estimate the size and create h5py dataset
        total_frames = self._get_total_frames(xys)

        patches_cache, labels_cache = self._create_patch_and_label_dataset(
            total_frames)

        assert 'dis_y' in xkeys
        assert 'dis_u' in xkeys
        assert 'dis_v' in xkeys

        yss = xys['dis_y'] # yss: Y * frames * videos
        uss = xys['dis_u']
        vss = xys['dis_v']
        if mode == 'train':
            assert 'label' in xys
            labels = xys['label']
        elif mode == 'test':
            labels = [None for _ in range(len(yss))]
        else:
            assert False

        assert len(yss) == len(uss) == len(vss) == len(labels)

        patch_idx = 0
        for ys, us, vs, label in zip(yss, uss, vss, labels): # iterate videos
            assert len(ys) == len(us) == len(vs)
            for y, u, v in zip(ys, us, vs): # iterate frames

                yuvimg = dstack_y_u_v(y, u, v)

                img = create_hp_yuv_4channel(yuvimg)

                h, w, c = img.shape

                adj_h = h - self.patch_height
                adj_w = w - self.patch_width

                iv, jv = np.meshgrid(np.arange(adj_h), np.arange(adj_w),
                                     sparse=False, indexing='ij')
                iv = iv.reshape(-1)
                jv = jv.reshape(-1)

                idx = np.random.permutation(adj_h * adj_w)

                iv = iv[idx]
                jv = jv[idx]

                patches_found = 0
                for (y, x) in zip(iv, jv):
                    patches_cache[patch_idx] = img[y: y + self.patch_height,
                                                   x: x + self.patch_width]
                    if mode == 'train':
                        labels_cache[patch_idx] = label

                    patches_found += 1
                    patch_idx += 1
                    if patches_found >= self.patches_per_frame:
                        break

        return patches_cache, labels_cache

    def _populate_patches(self, xkeys, xs):
        # reuse _populate_patches_and_labels to do the job
        patches_cache, _ = self._populate_patches_and_labels(
            xkeys, xys=xs, mode='test')
        return patches_cache

    def _get_total_frames(self, xys):
        yss = xys['dis_y'] # yss
        return np.sum(list(map(lambda ys:len(ys), yss)))

    @override(TrainTestModel)
    def to_file(self, filename, **more):

        self._assert_trained()

        # special handling for tensorflow: save .model differently
        model_dict_copy = self.model_dict.copy()
        model_dict_copy['model'] = None
        info_to_save = {'param_dict': self.param_dict,
                        'model_dict': model_dict_copy}

        saver = tf.train.Saver()
        sess = self.model['sess']
        saver.save(sess, filename + '.model')

        with open(filename, 'wb') as file:
            pickle.dump(info_to_save, file)

    @classmethod
    @override(TrainTestModel)
    def from_file(cls, filename, logger=None, optional_dict2=None, **more):
        format = more['format'] if 'format' in more else 'pkl'
        assert format in ['pkl'], f'format must be pkl but got {format}'

        assert os.path.exists(filename), 'File name {} does not exist.'.format(filename)
        with open(filename, 'rb') as file:
            info_loaded = pickle.load(file)

        train_test_model = cls(param_dict={}, logger=logger,
                               optional_dict2=optional_dict2)
        train_test_model.param_dict = info_loaded['param_dict']
        train_test_model.model_dict = info_loaded['model_dict']

        # == special handling of tensorflow: load .model differently ==

        input_image_batch, logits, y_, y_p, W_conv0, W_conv1, loss, train_step \
            = cls.create_tf_variables(train_test_model.param_dict)

        saver = tf.train.Saver()
        sess = tf.Session()
        checkpoint_dir = get_dir_without_last_slash(filename)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            assert False
        model = {
            'sess': sess,
            'y_p': y_p,
            'input_image_batch': input_image_batch,
        }
        train_test_model.model_dict['model'] = model

        return train_test_model

    @staticmethod
    @override(TrainTestModel)
    def delete(filename, **more):
        format = more['format'] if 'format' in more else 'pkl'
        assert format in ['pkl'], f'format must be pkl but got {format}'

        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists(filename + '.model'):
            os.remove(filename + '.model')
        if os.path.exists(filename + '.model.meta'):
            os.remove(filename + '.model.meta')
        filedir = get_dir_without_last_slash(filename)
        if os.path.exists(filedir + '/checkpoint'):
            os.remove(filedir + '/checkpoint')

    @classmethod
    def reset(cls):
        super(NeuralNetTrainTestModel, cls).reset()

        # reset tensorflow to avoid any memory
        tf.reset_default_graph()


class ToddNoiseClassifierTrainTestModel(NeuralNetTrainTestModel, ClassifierMixin):

    TYPE = "TODDNOISECLASSIFIER"
    VERSION = "0.1"

    # override NeuralNetTrainTestModel.DEFAULT_PATCHES_PER_FRAME
    DEFAULT_PATCHES_PER_FRAME = 10

    n_channels = 4
    n_filters = 5
    fsize0 = 5
    fsize1 = 3

    def _train(self, patches, labels):

        assert len(patches) == len(labels)

        # randomly split data into training and validation set
        num_data = len(patches)
        indices = np.random.permutation(num_data)
        num_train_data = int(num_data / 2) # do even split
        train_indices = indices[:num_train_data]
        validate_indices = indices[num_train_data:]
        train_posindices = list(filter(lambda idx: labels[idx] == 1, train_indices))
        train_negindices = list(filter(lambda idx: labels[idx] == 0, train_indices))

        input_image_batch, logits, y_, y_p, W_conv0, W_conv1, loss, train_step \
            = self.create_tf_variables(self.param_dict)

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        saver = None

        # loop through and compute training/testing loss per epoch
        f1score_per_epoch = []
        loss_per_epoch = []
        for j in range(self.n_epochs):
            print("")

            print("******************** EPOCH %d / %d ********************" % (j, self.n_epochs))

            # train
            train_loss, train_score = self._evaluate_on_patches(
                patches, labels, input_image_batch, loss,
                sess, train_indices, y_, y_p, "train")

            print("")

            # validate
            validate_loss, validate_score = self._evaluate_on_patches(
                patches, labels, input_image_batch, loss,
                sess, validate_indices, y_, y_p, "validate")

            print("")

            print("f1 train %g, f1 validate %g, loss train %g, loss validate %g"
                  % (train_score, validate_score, train_loss, validate_loss))
            f1score_per_epoch.append([train_score, validate_score])
            loss_per_epoch.append([train_loss, validate_loss])

            if self.checkpoints_dir:
                if saver is None:
                    saver = tf.train.Saver(max_to_keep=0)
                outputfile = "%s/model_epoch_%d.ckpt" % (self.checkpoints_dir, j,)
                print("Checkpointing -> %s" % (outputfile,))
                saver.save(sess, outputfile)

            halfbatch = self.batch_size // 2

            # here, we enforce balanced training, so that if we would like to
            # use hinge loss, we will always have representatives from both
            # target classes

            n_iterations = np.min((len(train_posindices) // halfbatch,
                                   len(train_negindices) // halfbatch))

            for i in range(n_iterations):
                # must sort, since h5py needs ordered indices
                poslst = np.sort(train_posindices[i*(halfbatch):(i+1)*(halfbatch)]).tolist()
                neglst = np.sort(train_negindices[i*(halfbatch):(i+1)*(halfbatch)]).tolist()

                X_batch = np.vstack((
                    patches[poslst],
                    patches[neglst],
                ))
                y_batch = np.vstack((
                    as_one_hot(labels[poslst]),
                    as_one_hot(labels[neglst])
                ))

                sys.stdout.write("Training: %d / %d\r" % (i, n_iterations))
                sys.stdout.flush()

                sess.run(train_step, feed_dict={input_image_batch: X_batch, y_: y_batch})

            np.random.shuffle(train_posindices)
            np.random.shuffle(train_negindices)

            print("")

        model = {
            'sess': sess,
            'y_p': y_p,
            'input_image_batch': input_image_batch,
        }

        return model

    def _evaluate_on_patches(self, patches_in, labels_in, input_image_batch,
                             loss_in, sess, indices, y_, y_p, type_="train"):
        ys_pred = []
        ys_true = []
        loss_cum = 0.0
        n_steps = len(indices) // self.batch_size
        for M in range(n_steps):
            sys.stdout.write("Evaluating {type} data: {M} / {nstep}\r"
                             .format(type=type_, M=M, nstep=n_steps))
            sys.stdout.flush()
            curr_indices = indices[M * self.batch_size:
                           (M + 1) * self.batch_size].tolist()
            patches = list(map(lambda idx: patches_in[idx], curr_indices))
            labels_ = list(map(lambda idx: labels_in[idx], curr_indices))
            labels = as_one_hot(labels_)
            y_pred, loss = sess.run([y_p, loss_in],
                feed_dict={input_image_batch: patches, y_: labels})
            ys_pred = np.hstack((ys_pred, y_pred))
            ys_true = np.hstack((ys_true, labels_))
            loss_cum += loss
        loss_cum /= n_steps
        score = f1_score(ys_true, ys_pred)
        return loss_cum, score

    @classmethod
    def _predict(cls, model, patches):

        sess = model['sess']
        y_p = model['y_p']
        input_image_batch = model['input_image_batch']

        ys_patche_pred = sess.run(y_p, feed_dict={input_image_batch: patches})

        # predict by majority voting on all patches
        y_pred = scipy.stats.mode(ys_patche_pred)[0][0]

        return y_pred

    @classmethod
    def create_tf_variables(cls, param_dict):
        def weight_variable(shape, name="test", seed=None):
            return tf.Variable(tf.random_normal(shape, stddev=0.1, seed=seed), name=name)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

        if param_dict is not None and 'patch_width' in param_dict:
            patch_width = param_dict['patch_width']
        else:
            patch_width = cls.DEFAULT_PATCH_WIDTH

        if param_dict is not None and 'patch_height' in param_dict:
            patch_height = param_dict['patch_height']
        else:
            patch_height = cls.DEFAULT_PATCH_HEIGHT

        if param_dict is not None and 'learning_rate' in param_dict:
            learning_rate = param_dict['learning_rate']
        else:
            learning_rate = cls.DEFAULT_LEARNING_RATE

        if param_dict is not None and 'seed' in param_dict:
            seed = param_dict['seed']
        else:
            seed = cls.DEFAULT_SEED

        tf.set_random_seed(seed)

        n_channels = cls.n_channels
        response_map_size_width = patch_width - (cls.fsize0 // 2) * 2
        response_map_size_width -= (cls.fsize1 // 2) * 2
        response_map_size_height = patch_height - (cls.fsize0 // 2) * 2
        response_map_size_height -= (cls.fsize1 // 2) * 2
        response_map_size = response_map_size_height * response_map_size_width
        input_image_batch = tf.placeholder(
            tf.float32, shape=[None, patch_height, patch_width, n_channels])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])
        W_conv0 = weight_variable(
            [cls.fsize0, cls.fsize0, n_channels, cls.n_filters], "W_conv0")
        W_conv1 = weight_variable(
            [cls.fsize1, cls.fsize1, cls.n_filters, 2], "W_conv1")

        # convolutional layer 1
        h_conv0 = conv2d(input_image_batch, W_conv0)
        h_conv0_elu = tf.nn.elu(h_conv0)

        # layer 2, which is just an output layer
        h_conv1 = conv2d(h_conv0_elu, W_conv1)
        h_conv1_elu = tf.nn.elu(h_conv1)
        h_conv1_elu_flat = tf.reshape(h_conv1_elu, [-1, response_map_size, 2])

        logits = tf.reduce_mean(h_conv1_elu_flat, 1)
        logits_norm = tf.nn.softmax(logits)

        y_p = tf.argmax(logits_norm, 1)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        return input_image_batch, logits, y_, y_p, W_conv0, W_conv1, loss, train_step
