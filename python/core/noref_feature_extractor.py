__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
from time import sleep
import numpy as np
from scipy.misc import imresize
from scipy.special import gamma
from scipy.ndimage import correlate1d
import multiprocessing

from tools.misc import make_parent_dirs_if_nonexist, get_dir_without_last_slash
from core.feature_extractor import FeatureExtractor
from tools.reader import YuvReader

class NorefFeatureExtractor(FeatureExtractor):
    """
    NorefFeatureExtractor is a subclass of FeatureExtractor. Any derived
    subclass of NorefFeatureExtractor cannot use the reference video of an
    input asset.
    """

    def _assert_paths(self, asset):
        # Override Executor._assert_paths to skip asserting on ref_path
        assert os.path.exists(asset.ref_path), \
            "Distorted path {} does not exist.".format(asset.dis_path)

    def _wait_for_workfiles(self, asset):
        # Override Executor._wait_for_workfiles to skip ref_workfile_path
        # wait til workfile paths being generated
        # FIXME: use proper mutex (?)
        for i in range(10):
            if os.path.exists(asset.dis_workfile_path):
                break
            sleep(0.1)
        else:
            raise RuntimeError("dis video workfile path {} is missing.".format(
                asset.dis_workfile_path))

    def _run_on_asset(self, asset):
        # Override Executor._run_on_asset to skip working on ref video

        # asserts
        self._assert_an_asset(asset)

        if self.result_store:
            result = self.result_store.load(asset, self.executor_id)
        else:
            result = None

        # if result can be retrieved from result_store, skip log file
        # generation and reading result from log file, but directly return
        # return the retrieved result
        if result is not None:
            if self.logger:
                self.logger.info('{id} result exists. Skip {id} run.'.
                                 format(id=self.executor_id))
        else:

            if self.logger:
                self.logger.info('{id} result does\'t exist. Perform {id} '
                                 'calculation.'.format(id=self.executor_id))

            # at this stage, it is certain that asset.ref_path and
            # asset.dis_path will be used. must early determine that
            # they exists
            self._assert_paths(asset)

            # if no rescaling is involved, directly work on ref_path/dis_path,
            # instead of opening workfiles
            self._set_asset_use_path_as_workpath(asset)

            # remove workfiles if exist (do early here to avoid race condition
            # when ref path and dis path have some overlap)
            if asset.use_path_as_workpath:
                # do nothing
                pass
            else:
                self._close_dis_workfile(asset)

            log_file_path = self._get_log_file_path(asset)
            make_parent_dirs_if_nonexist(log_file_path)

            if asset.use_path_as_workpath:
                # do nothing
                pass
            else:
                if self.fifo_mode:
                    dis_p = multiprocessing.Process(target=self._open_dis_workfile,
                                                    args=(asset, True))
                    dis_p.start()
                    self._wait_for_workfiles(asset)
                else:
                    self._open_dis_workfile(asset, fifo_mode=False)

            self._prepare_log_file(asset)

            self._run_and_generate_log_file(asset)

            # clean up workfiles
            if self.delete_workdir:
                if asset.use_path_as_workpath:
                    # do nothing
                    pass
                else:
                    self._close_dis_workfile(asset)

            if self.logger:
                self.logger.info("Read {id} log file, get scores...".
                                 format(type=self.executor_id))

            # collect result from each asset's log file
            result = self._read_result(asset)

            # save result
            if self.result_store:
                self.result_store.save(result)

            # clean up workdir and log files in it
            if self.delete_workdir:

                # remove log file
                self._remove_log(asset)

                # remove dir
                log_file_path = self._get_log_file_path(asset)
                log_dir = get_dir_without_last_slash(log_file_path)
                try:
                    os.rmdir(log_dir)
                except OSError as e:
                    if e.errno == 39: # [Errno 39] Directory not empty
                        # VQM could generate an error file with non-critical
                        # information like: '3 File is longer than 15 seconds.
                        # Results will be calculated using first 15 seconds
                        # only.' In this case, want to keep this
                        # informational file and pass
                        pass

        result = self._post_process_result(result)

        return result

class MomentNorefFeatureExtractor(NorefFeatureExtractor):

    TYPE = "Moment_noref_feature"
    VERSION = "1.0" # python only

    ATOM_FEATURES = ['1st', '2nd', ] # order matters

    DERIVED_ATOM_FEATURES = ['var', ]

    def _run_and_generate_log_file(self, asset):
        # routine to call the command-line executable and generate feature
        # scores in the log file.

        quality_w, quality_h = asset.quality_width_height
        with YuvReader(filepath=asset.dis_workfile_path, width=quality_w,
                       height=quality_h, yuv_type=asset.yuv_type) as dis_yuv_reader:
            scores_mtx_list = []
            i = 0
            for dis_yuv in dis_yuv_reader:
                dis_y = dis_yuv[0]
                firstm = dis_y.mean()
                secondm = dis_y.var() + firstm**2
                scores_mtx_list.append(np.hstack(([firstm], [secondm])))
                i += 1
            scores_mtx = np.vstack(scores_mtx_list)

        # write scores_mtx to log file
        log_file_path = self._get_log_file_path(asset)
        with open(log_file_path, "wb") as log_file:
            np.save(log_file, scores_mtx)

    def _get_feature_scores(self, asset):
        # routine to read the feature scores from the log file, and return
        # the scores in a dictionary format.

        log_file_path = self._get_log_file_path(asset)
        with open(log_file_path, "rb") as log_file:
            scores_mtx = np.load(log_file)

        num_frm, num_features = scores_mtx.shape
        assert num_features == len(self.ATOM_FEATURES)

        feature_result = {}

        for idx, atom_feature in enumerate(self.ATOM_FEATURES):
            scores_key = self.get_scores_key(atom_feature)
            feature_result[scores_key] = list(scores_mtx[:, idx])

        return feature_result

    @classmethod
    def _post_process_result(cls, result):
        # override Executor._post_process_result(result)

        result = super(MomentNorefFeatureExtractor, cls)._post_process_result(result)

        # calculate var from 1st, 2nd
        var_scores_key = cls.get_scores_key('var')
        first_scores_key = cls.get_scores_key('1st')
        second_scores_key = cls.get_scores_key('2nd')
        get_var = lambda (m1, m2): m2 - m1 * m1
        result.result_dict[var_scores_key] = \
            map(get_var, zip(result.result_dict[first_scores_key],
                             result.result_dict[second_scores_key]))

        # validate
        for feature in cls.DERIVED_ATOM_FEATURES:
            assert cls.get_scores_key(feature) in result.result_dict

        return result

class BrisqueNorefFeatureExtractor(NorefFeatureExtractor):

    TYPE = "BRISQUE_noref_feature"

    # VERSION = "0.1"
    VERSION = "0.2" # update PIL package to 3.2 to fix interpolation issue 

    ATOM_FEATURES = [
        "alpha_m1", "sq_m1",
        "alpha_m2", "sq_m2",
        "alpha_m3", "sq_m3",

        "alpha11", "N11", "lsq11", "rsq11",
        "alpha12", "N12", "lsq12", "rsq12",
        "alpha13", "N13", "lsq13", "rsq13",
        "alpha14", "N14", "lsq14", "rsq14",

        "alpha21", "N21", "lsq21", "rsq21",
        "alpha22", "N22", "lsq22", "rsq22",
        "alpha23", "N23", "lsq23", "rsq23",
        "alpha24", "N24", "lsq24", "rsq24",

        "alpha31", "N31", "lsq31", "rsq31",
        "alpha32", "N32", "lsq32", "rsq32",
        "alpha33", "N33", "lsq33", "rsq33",
        "alpha34", "N34", "lsq34", "rsq34",
    ] # order matters

    gamma_range = np.arange(0.2, 10, 0.001)
    a = gamma(2.0 / gamma_range)**2
    b = gamma(1.0 / gamma_range)
    c = gamma(3.0 / gamma_range)
    prec_gammas = a / (b * c)

    def _run_and_generate_log_file(self, asset):
        # routine to call the command-line executable and generate feature
        # scores in the log file.

        quality_w, quality_h = asset.quality_width_height
        with YuvReader(filepath=asset.dis_workfile_path, width=quality_w,
                       height=quality_h, yuv_type=asset.yuv_type) as dis_yuv_reader:
            scores_mtx_list = []
            i = 0
            for dis_yuv in dis_yuv_reader:
                dis_y = dis_yuv[0]
                fgroup1_dis, fgroup2_dis = self.mscn_extract(dis_y)
                scores_mtx_list.append(np.hstack((fgroup1_dis, fgroup2_dis)))
                i += 1
            scores_mtx = np.vstack(scores_mtx_list)

        # write scores_mtx to log file
        log_file_path = self._get_log_file_path(asset)
        with open(log_file_path, "wb") as log_file:
            np.save(log_file, scores_mtx)

    def _get_feature_scores(self, asset):
        # routine to read the feature scores from the log file, and return
        # the scores in a dictionary format.

        log_file_path = self._get_log_file_path(asset)
        with open(log_file_path, "rb") as log_file:
            scores_mtx = np.load(log_file)

        num_frm, num_features = scores_mtx.shape
        assert num_features == len(self.ATOM_FEATURES)

        feature_result = {}

        for idx, atom_feature in enumerate(self.ATOM_FEATURES):
            scores_key = self.get_scores_key(atom_feature)
            feature_result[scores_key] = list(scores_mtx[:, idx])

        return feature_result

    @classmethod
    def mscn_extract(cls, img):
        img2 = imresize(img, 0.5)
        img3 = imresize(img, 0.25)
        m_image, _, _ = cls.calc_image(img)
        m_image2, _, _ = cls.calc_image(img2)
        m_image3, _, _ = cls.calc_image(img3)

        pps11, pps12, pps13, pps14 = cls.paired_p(m_image)
        pps21, pps22, pps23, pps24 = cls.paired_p(m_image2)
        pps31, pps32, pps33, pps34 = cls.paired_p(m_image3)

        alpha11, N11, bl11, br11, lsq11, rsq11 = cls.extract_aggd_features(pps11)
        alpha12, N12, bl12, br12, lsq12, rsq12 = cls.extract_aggd_features(pps12)
        alpha13, N13, bl13, br13, lsq13, rsq13 = cls.extract_aggd_features(pps13)
        alpha14, N14, bl14, br14, lsq14, rsq14 = cls.extract_aggd_features(pps14)

        alpha21, N21, bl21, br21, lsq21, rsq21 = cls.extract_aggd_features(pps21)
        alpha22, N22, bl22, br22, lsq22, rsq22 = cls.extract_aggd_features(pps22)
        alpha23, N23, bl23, br23, lsq23, rsq23 = cls.extract_aggd_features(pps23)
        alpha24, N24, bl24, br24, lsq24, rsq24 = cls.extract_aggd_features(pps24)

        alpha31, N31, bl31, br31, lsq31, rsq31 = cls.extract_aggd_features(pps31)
        alpha32, N32, bl32, br32, lsq32, rsq32 = cls.extract_aggd_features(pps32)
        alpha33, N33, bl33, br33, lsq33, rsq33 = cls.extract_aggd_features(pps33)
        alpha34, N34, bl34, br34, lsq34, rsq34 = cls.extract_aggd_features(pps34)

        alpha_m1, sq_m1 = cls.extract_ggd_features(m_image)
        alpha_m2, sq_m2 = cls.extract_ggd_features(m_image2)
        alpha_m3, sq_m3 = cls.extract_ggd_features(m_image3)

        mscn_features = np.array([
                alpha_m1, sq_m1, #0, 1
                alpha_m2, sq_m2, #0, 1
                alpha_m3, sq_m3, #0, 1
        ])

        pp_features = np.array([
                alpha11, N11, lsq11, rsq11, #6, 7, 8, 9 (V)
                alpha12, N12, lsq12, rsq12, #10, 11, 12, 13 (H)
                alpha13, N13, lsq13, rsq13, #14, 15, 16, 17 (D1)
                alpha14, N14, lsq14, rsq14, #18, 19, 20, 21 (D2)
                alpha21, N21, lsq21, rsq21, #6, 7, 8, 9 (V)
                alpha22, N22, lsq22, rsq22, #10, 11, 12, 13 (H)
                alpha23, N23, lsq23, rsq23, #14, 15, 16, 17 (D1)
                alpha24, N24, lsq24, rsq24, #18, 19, 20, 21 (D2)
                alpha31, N31, lsq31, rsq31, #6, 7, 8, 9 (V)
                alpha32, N32, lsq32, rsq32, #10, 11, 12, 13 (H)
                alpha33, N33, lsq33, rsq33, #14, 15, 16, 17 (D1)
                alpha34, N34, lsq34, rsq34, #18, 19, 20, 21 (D2)
        ])

        return mscn_features, pp_features

    @staticmethod
    def gauss_window(lw, sigma):
        sd = float(sigma)
        lw = int(lw)
        weights = [0.0] * (2 * lw + 1)
        weights[lw] = 1.0
        sum = 1.0
        sd *= sd
        for ii in range(1, lw + 1):
            tmp = np.exp(-0.5 * float(ii * ii) / sd)
            weights[lw + ii] = tmp
            weights[lw - ii] = tmp
            sum += 2.0 * tmp
        for ii in range(2 * lw + 1):
            weights[ii] /= sum
        return weights

    @classmethod
    def calc_image(cls, image):
        extend_mode = 'constant' #'nearest' #'wrap'
        avg_window = cls.gauss_window(3, 7.0/6.0)
        w, h = np.shape(image)
        mu_image = np.zeros((w, h))
        var_image = np.zeros((w, h))
        image = np.array(image).astype('float')
        correlate1d(image,     avg_window, 0, mu_image,  mode=extend_mode)
        correlate1d(mu_image,  avg_window, 1, mu_image,  mode=extend_mode)
        correlate1d(image**2,  avg_window, 0, var_image, mode=extend_mode)
        correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
        var_image = np.sqrt(np.abs(var_image - mu_image**2))
        return (image - mu_image)/(var_image + 1), var_image, mu_image

    @staticmethod
    def paired_p(new_im):
        hr_shift = np.roll(new_im,   1, axis=1)
        hl_shift = np.roll(new_im,  -1, axis=1)
        v_shift  = np.roll(new_im,   1, axis=0)
        vr_shift = np.roll(hr_shift, 1, axis=0)
        vl_shift = np.roll(hl_shift, 1, axis=0)

        D1_img = vr_shift * new_im
        D2_img = vl_shift * new_im
        H_img  = hr_shift * new_im
        V_img  = v_shift  * new_im

        return V_img, H_img, D1_img, D2_img

    @classmethod
    def extract_ggd_features(cls, imdata):
        nr_gam = 1.0 / cls.prec_gammas
        sigma_sq = np.average(imdata**2)
        E = np.average(np.abs(imdata))
        rho = sigma_sq / E**2
        pos = np.argmin(np.abs(nr_gam - rho))
        return cls.gamma_range[pos], np.sqrt(sigma_sq)

    @classmethod
    def extract_aggd_features(cls, imdata):
        # flatten imdata
        imdata.shape = (len(imdata.flat),)
        imdata2 = imdata*imdata
        left_data = imdata2[imdata<0]
        right_data = imdata2[imdata>=0]
        left_mean_sqrt = 0
        right_mean_sqrt = 0
        if len(left_data) > 0:
            left_mean_sqrt = np.sqrt(np.average(left_data))
        if len(right_data) > 0:
            right_mean_sqrt = np.sqrt(np.average(right_data))

        gamma_hat = left_mean_sqrt/right_mean_sqrt
        # solve r-hat norm
        r_hat = (np.average(np.abs(imdata))**2) / (np.average(imdata2))
        rhat_norm = r_hat * (((gamma_hat**3 + 1) * (gamma_hat + 1))
                             / ((gamma_hat**2 + 1)**2))

        # solve alpha by guessing values that minimize ro
        pos = np.argmin(np.abs(cls.prec_gammas - rhat_norm))
        alpha = cls.gamma_range[pos]

        gam1 = gamma(1.0/alpha)
        gam2 = gamma(2.0/alpha)
        gam3 = gamma(3.0/alpha)

        aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
        bl = aggdratio * left_mean_sqrt
        br = aggdratio * right_mean_sqrt

        # mean parameter
        N = (br - bl) * (gam2 / gam1) * aggdratio
        return alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt
