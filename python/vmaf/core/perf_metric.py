from abc import abstractmethod, ABCMeta
import numpy as np
from numpy.linalg import lstsq
import scipy.stats
import scipy.special
import scipy.interpolate

from vmaf.core.mixin import TypeVersionEnabled
from vmaf.tools.decorator import override
from vmaf.tools.misc import empty_object, indices
from vmaf.tools.sigproc import fastDeLong, calpvalue, significanceHM, \
    significanceBinomial

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class PerfMetric(TypeVersionEnabled):

    __metaclass__ = ABCMeta

    @classmethod
    @abstractmethod
    def _preprocess(cls, groundtruths, predictions, **kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _evaluate(cls, groundtruths, predictions, **kwargs):
        raise NotImplementedError

    def __init__(self, groundtruths, predictions):
        """
        Performance metric on quality metrics
        :param groundtruths: either list of real numbers (aggregate scores like
        MOS or DMOS or MLE), or list of list of real numbers (list of raw scores)
        :param predictions: list of real numbers
        :return:
        """
        TypeVersionEnabled.__init__(self)
        self.groundtruths = groundtruths
        self.predictions = predictions
        self._assert_args()

    def _assert_args(self):
        assert len(self.groundtruths) == len(self.predictions), 'The lengths of groundtruth labels and predictions do not match.'

    def evaluate(self, **kwargs):
        """
        :return: ret - a dictionary with 'score' and other keys
        """
        groundtruths, predictions = self._preprocess(self.groundtruths, self.predictions, **kwargs)
        result = self._evaluate(groundtruths, predictions, **kwargs)
        assert 'score' in result, 'Score does not exist in result.'
        return result


class RawScorePerfMetric(PerfMetric):
    """
    Groundtruth is a list of raw scores (list of list of real numbers)
    """
    @override(PerfMetric)
    def _assert_args(self):
        super(RawScorePerfMetric, self)._assert_args()

        # require the raw scores to be more than 1
        for groundtruth in self.groundtruths:
            assert hasattr(groundtruth, '__len__') and len(groundtruth) > 1


class AucPerfMetric(RawScorePerfMetric):
    """
    # % The method is described in the paper:
    # % L. Krasula, K. Fliegel, P. Le Callet, M.Klima, "On the accuracy of
    # % objective image and video quality models: New methodology for
    # % performance evaluation", QoMEX 2016.
    # % When you use our method in your research, please, cite the above stated
    # % paper.
    # %
    # % Copyright (c) 2016
    # % Lukas Krasula <l.krasula@gmail.com>
    #
    # % Permission to use, copy, modify, and/or distribute this software for any
    # % purpose with or without fee is hereby granted, provided that the above
    # % copyright notice and this permission notice appear in all copies.
    # %
    # % THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
    # % WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
    # % MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR
    # % ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
    # % WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
    # % ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
    # % OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
    # %
    # % This software also uses the code described in:
    # % X. Sun and W. Xu, "Fast Implementation of DeLong's Algorithm for
    # % Comparing the Areas Under Correlated Receiver Operating Characteristic
    # % Curves," IEEE Signal Processing Letters, vol. 21, no. 11, pp. 1389-1393,
    # % 2014.
    # %
    # % and
    # %
    # % P. Hanhart, L. Krasula, P. Le Callet, T. Ebrahimi, "How to benchmark
    # % objective quality metrics from pair comparison data", QoMEX 2016.
    """

    TYPE = "AUC"
    VERSION = "0.1"

    @classmethod
    @override(PerfMetric)
    def _preprocess(cls, groundtruths, predictions, **kwargs):
        return groundtruths, predictions

    @staticmethod
    def _metrics_performance(objScoDif, signif):
        """
        mirroring matlab function:
        %[results] = Metrics_performance(objScoDif, signif, doPlot)
        % INPUT:    objScoDif   : differences of objective scores [M,N]
        %                         M : number of metrics
        %                         N : number of pairs
        %           signif      : statistical outcome of paired comparison [1,N]
        %                          0 : no difference
        %                         -1 : first stimulus is worse
        %                          1 : first stimulus is better
        %           doPlot      : boolean indicating if graphs should be plotted
        %
        % OUTPUT:   results - structure with following fields
        %
        %           AUC_DS      : Area Under the Curve for Different/Similar ROC
        %                         analysis
        %           pDS_DL      : p-values for AUC_DS from DeLong test
        %           pDS_HM      : p-values for AUC_DS from Hanley and McNeil test
        %           AUC_BW      : Area Under the Curve for Better/Worse ROC
        %                         analysis
        %           pBW_DL      : p-values for AUC_BW from DeLong test
        %           pBW_HM      : p-values for AUC_BW from Hanley and McNeil test
        %           CC_0        : Correct Classification @ DeltaOM = 0 for
        %                         Better/Worse ROC analysis
        %           pCC0_b      : p-values for CC_0 from binomial test
        %           pCC0_F      : p-values for CC_0 from Fisher's exact test
        %           THR         : threshold for 95% probability that the stimuli
        %                         are different
        """

        # M = size(objScoDif,1);
        # D = abs(objScoDif(:,signif ~= 0));
        # S = abs(objScoDif(:,signif == 0));
        # samples.spsizes = [size(D,2),size(S,2)];
        # samples.ratings = [D,S];

        M = objScoDif.shape[0]
        D = np.abs(objScoDif[:, indices(signif[0], lambda x: x != 0)])
        S = np.abs(objScoDif[:, indices(signif[0], lambda x: x == 0)])
        samples = empty_object()
        samples.spsizes = [D.shape[1], S.shape[1]]
        samples.ratings = np.hstack([D, S])

        # % calculate AUCs

        # [AUC_DS,C] = fastDeLong(samples);
        AUC_DS, C, _, _ = fastDeLong(samples)

        # % significance calculation

        # pDS_DL = ones(M);
        # for i=1:M-1
        #     for j=i+1:M
        #         pDS_DL(i,j) = calpvalue(AUC_DS([i,j]), C([i,j],[i,j]));
        #         pDS_DL(j,i) = pDS_DL(i,j);
        #     end
        # end
        pDS_DL = np.ones([M, M])
        for i in range(1, M):
            for j in range(i + 1, M + 1):
                # http://stackoverflow.com/questions/4257394/slicing-of-a-numpy-2d-array-or-how-do-i-extract-an-mxm-submatrix-from-an-nxn-ar
                pDS_DL[i - 1, j - 1] = calpvalue(AUC_DS[[i - 1, j - 1]], C[[[i - 1], [j - 1]], [i - 1, j - 1]])
                pDS_DL[j - 1, i - 1] = pDS_DL[i - 1, j - 1]

        ## [pDS_HM,CI_DS] = significanceHM(S, D, AUC_DS);
        # pDS_HM, CI_DS = significanceHM(S, D, AUC_DS)

        # THR = prctile(D',95);
        THR = np.percentile(S, 95, axis=1)

        # %%%%%%%%%%%%%%%%%%%%%%% Better / Worse %%%%%%%%%%%%%%%%%%%%%%%%%%%

        # B = [objScoDif(:,signif == 1),-objScoDif(:,signif == -1)];
        # W = -B;
        # samples.ratings = [B,W];
        # samples.spsizes = [size(B,2),size(W,2)];
        B1 = objScoDif[:, indices(signif[0], lambda x: x == 1)]
        B2 = objScoDif[:, indices(signif[0], lambda x: x == -1)]
        B = np.hstack([B1, -B2])
        W = -B
        samples = empty_object()
        samples.ratings = np.hstack([B, W])
        samples.spsizes = [B.shape[1], W.shape[1]]

        # % calculate AUCs

        # [AUC_BW,C] = fastDeLong(samples);
        AUC_BW, C, _, _ = fastDeLong(samples)

        # % calculate correct classification for DeltaOM = 0

        # L = size(B,2) + size(W,2);
        # CC_0 = zeros(M,1);
        # for m=1:M
        #     CC_0(m) = (sum(B(m,:)>0) + sum(W(m,:)<0)) / L;
        # end
        L = B.shape[1] + W.shape[1]
        CC_0 = np.zeros(M)
        for m in range(M):
            CC_0[m] = float(np.sum(B[m, :] > 0) + np.sum(W[m, :] < 0)) / L

        # % significance calculation

        # pBW_DL = ones(M);
        # pCC0_b = ones(M);
        # pCC0_F = ones(M);
        # for i=1:M-1
        #     for j=i+1:M
        #         pBW_DL(i,j) = calpvalue(AUC_BW([i,j]), C([i,j],[i,j]));
        #         pBW_DL(j,i) = pBW_DL(i,j);
        #
        #         pCC0_b(i,j) = significanceBinomial(CC_0(i), CC_0(j), L);
        #         pCC0_b(j,i) = pCC0_b(i,j);
        #
        #         pCC0_F(i,j) = fexact(CC_0(i)*L, 2*L, CC_0(i)*L + CC_0(j)*L, L, 'tail', 'b')/2;
        #         pCC0_F(j,i) = pCC0_F(i,j);
        #     end
        # end
        pBW_DL = np.ones([M, M])
        pCC0_b = np.ones([M, M])
        # pCC0_F = np.ones([M, M])
        for i in range(1, M):
            for j in range(i + 1, M + 1):
                pBW_DL[i - 1, j - 1] = calpvalue(AUC_BW[[i - 1, j - 1]], C[[[i - 1], [j - 1]], [i - 1, j - 1]])
                pBW_DL[j - 1, i - 1] = pBW_DL[i - 1, j - 1]

                pCC0_b[i - 1, j - 1] = significanceBinomial(CC_0[i - 1], CC_0[j - 1], L)
                pCC0_b[j - 1, i - 1] = pCC0_b[i - 1, j - 1]

                # pCC0_F[i-1, j-1] = fexact(CC_0[i-1]*L, 2*L, CC_0[i-1]*L + CC_0[j-1]*L, L, 'tail', 'b') / 2.0
                # pCC0_F[j-1, i-1] = pCC0_F[i-1,j]

        # # [pBW_HM,CI_BW] = significanceHM(B, W, AUC_BW);
        # pBW_HM, CI_BW = significanceHM(B, W, AUC_BW)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # % Adding outputs to the structure

        # results.AUC_DS = AUC_DS;
        # results.pDS_DL = pDS_DL;
        # results.pDS_HM = pDS_HM;
        # results.AUC_BW = AUC_BW;
        # results.pBW_DL = pBW_DL;
        # results.pBW_HM = pBW_HM;
        # results.CC_0 = CC_0;
        # results.pCC0_b = pCC0_b;
        # results.pCC0_F = pCC0_F;
        # results.THR = THR;
        result = {
            'AUC_DS': AUC_DS,
            'pDS_DL': pDS_DL,
            # 'pDS_HM': pDS_HM,
            'AUC_BW': AUC_BW,
            'pBW_DL': pBW_DL,
            # 'pBW_HM': pBW_HM,
            'CC_0': CC_0,
            'pCC0_b': pCC0_b,
            # 'pCC0_F': pCC0_F,
            'THR': THR,
        }

        # %%%%%%%%%%%%%%%%%%%%%%%% Plot Results %%%%%%%%%%%%%%%%%%%%%%%%%%%
        #
        # if(doPlot == 1)
        #
        # % Using Benjamini-Hochberg procedure for multiple comparisons in plots
        # % (note: correlation between groups has to be positive)
        #
        # plot_auc(results.pDS_HM,results.AUC_DS, CI_DS, 'AUC (-)','Different/Similar')
        # plot_cc(results.pCC0_F,results.CC_0,'C_0 (%)','Better/Worse')
        # plot_auc(results.pBW_HM,results.AUC_BW, CI_BW, 'AUC (-)','Better/Worse')
        #
        # end

        return result

    @classmethod
    def _evaluate(cls, groundtruths, predictions, **kwargs):

        if isinstance(groundtruths, (list, tuple)) and isinstance(groundtruths[0], dict):
            raise TypeError("{} cannot handle dictionary-style daataset yet.".format(cls.__name__))

        def _signif(a, b):
            mos_a = np.mean(a)
            mos_b = np.mean(b)
            n_a = len(a)
            n_b = len(b)
            var_a = np.var(a, ddof=1)
            var_b = np.var(b, ddof=1)
            den = var_a / n_a + var_b / n_b
            if den == 0.0:
                den = 1e-8
            z = (mos_a - mos_b) / np.sqrt(den)
            if z < -2:
                return -1
            elif z > 2:
                return 1
            else:
                return 0

        # generate pairs
        N = len(groundtruths)
        signif_mtx = np.zeros([N, N])
        i = 0
        for groundtruth in groundtruths:
            j = 0
            for groundtruth2 in groundtruths:
                signif = _signif(groundtruth, groundtruth2)
                signif_mtx[i, j] = signif
                j += 1
            i += 1

        if isinstance(predictions[0], list):
            M = len(predictions)
        else:
            M = 1

        objscodif_all = np.zeros([M, N * N])
        for metric_idx in range(M):
            objscodif_mtx = np.zeros([N, N])

            if isinstance(predictions[0], list):
                metric_predictions = predictions[metric_idx]
            else:
                metric_predictions = predictions

            i = 0
            for prediction in metric_predictions:
                j = 0
                for prediction2 in metric_predictions:
                    objscodif = prediction - prediction2
                    objscodif_mtx[i, j] = objscodif
                    j += 1
                i += 1

            objscodif_all[metric_idx, :] = objscodif_mtx.reshape(1, N * N)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(objscodif_mtx, interpolation='nearest')
        # plt.set_cmap('gray')
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(signif_mtx, interpolation='nearest')
        # plt.set_cmap('gray')
        # plt.colorbar()
        # DisplayConfig.show()

        results = cls._metrics_performance(objscodif_all, signif_mtx.reshape(1, N * N))
        results['score'] = results['AUC_BW']

        if isinstance(predictions[0], list):
            return results
        else:
            result = {}
            for key in results:
                result[key] = results[key][0]
            return result

    def _assert_args(self):
        if isinstance(self.predictions[0], list):
            for metric in self.predictions:
                assert len(self.groundtruths) == len(metric), 'The lengths of groundtruth labels and predictions do not match.'
                for score in metric:
                    assert isinstance(score, float) or isinstance(score, int), 'Predictions need to be a list of lists of numbers.'

        else:
            assert len(self.groundtruths) == len(self.predictions), 'The lengths of groundtruth labels and predictions do not match.'

class ResolvingPowerPerfMetric(RawScorePerfMetric):
    """
    The method is described in the paper:
    M. H. Pinson, S. Wolf, "Techniques for Evaluating Objective Video Quality Models Using
    Overlapping Subjective Data Sets", NTIA Technical Report TR-09-457.
    """

    TYPE = "ResPow"
    VERSION = "0.1"

    @staticmethod
    def sigmoid_adjust_raw(xs, ys):
        ys_max = np.max(ys) + 0.1
        ys_min = np.min(ys) - 0.1

        ys_ = np.mean(np.array(ys), axis=1)

        # normalize to [0, 1]
        ys_ = (ys_ - ys_min) / (ys_max - ys_min)

        zs = -np.log(1.0 / ys_.T - 1.0)
        Y_mtx = np.array((np.ones(len(ys_)), zs)).T
        x_vec = np.array([xs]).T
        a_b = lstsq(Y_mtx, x_vec, rcond=-1)[0]
        a = a_b.item(0)
        b = a_b.item(1)

        xs = 1.0 / (1.0 + np.exp(- (np.array(xs) - a) / b))

        # denormalize
        xs = xs * (ys_max - ys_min) + ys_min

        return xs

    @classmethod
    @override(PerfMetric)
    def _preprocess(cls, groundtruths, predictions, **kwargs):
        enable_mapping = kwargs['enable_mapping'] if 'enable_mapping' in kwargs else False

        if enable_mapping:
            predictions_ = cls.sigmoid_adjust_raw(predictions, groundtruths)
        else:
            predictions_ = predictions

        return groundtruths, predictions_

    @classmethod
    def _evaluate(cls, groundtruths, predictions, **kwargs):

        # function [resolving_power] = vqm_accuracy (vqm, num_viewers, mos, std, deg_of_freedom) % MATLAB function [resolving_power] = ...
        # % vqm_accuracy (vqm, num_viewers, mos, std, deg_of_freedom)
        # %
        # % Compute resolving power for one model.
        # %
        # %    vqm is the video quality metric score for this src_id x hrc_id
        # %    num_viewers is the number of viewers that rated this src_id x hrc_id
        # %    mos is the mean opinion score of this src_id x hrc_id
        # %    std is the standard-deviation of this src_id x hrc_id
        # %
        #
        # % All of the above arrays must be the same length.  The VQM must already be
        # % fitted to the MOS.
        # %
        # %   deg_of_freedom is the number of degrees of freedom for the fit between
        # %           VQM and MOS prior to calling this routine.
        # %
        # % returned data contains:
        # %   resolving_power(1) = 95% Resolving Power
        # %   resolving_power(2) = 90% Resolving Power
        # %   resolving_power(3) = 75% Resolving Power
        # %   resolving_power(4) = 68% Resolving Power

        if isinstance(groundtruths, (list, tuple)) and isinstance(groundtruths[0], dict):
            raise TypeError("{} cannot handle dictionary-style daataset yet.".format(cls.__name__))

        deg_of_freedom = kwargs['ddof'] if 'ddof' in kwargs else 0

        vqm = np.array(predictions)
        num_viewers = np.array(list(map(lambda groundtruth: len(groundtruth), groundtruths)))
        mos = np.array(list(map(lambda groundtruth: np.nanmean(groundtruth), groundtruths)))
        std = np.array(list(map(lambda groundtruth: np.nanstd(groundtruth, ddof=deg_of_freedom), groundtruths)))

        # variance = std.^2;
        variance = std**2

        # num_comb = length(vqm);
        num_comb = len(vqm)

        # % Perform the vqm RMSE calculation using vqm.
        # vqm_rmse = (sum((vqm-mos).^2)/(num_comb - deg_of_freedom))^0.5;
        vqm_rmse = (sum((vqm-mos)**2)/(num_comb - deg_of_freedom))**0.5

        # % Perform the vqm resolution measurement using both vqm and mos.
        # vqm_pairs = repmat(vqm,1,num_comb)-repmat(vqm',num_comb,1);
        # mos_pairs = repmat(mos,1,num_comb)-repmat(mos',num_comb,1);
        # stand_err_diff = sqrt(repmat(variance./num_viewers,1,num_comb) + repmat((variance./num_viewers)',num_comb,1));
        # z_pairs = mos_pairs./stand_err_diff;
        vqm_pairs = np.tile(vqm, (num_comb, 1))
        vqm_pairs = vqm_pairs - vqm_pairs.T
        mos_pairs = np.tile(mos, (num_comb, 1))
        mos_pairs = mos_pairs - mos_pairs.T
        stand_err_diff = np.tile(variance / num_viewers, (num_comb, 1))
        stand_err_diff = np.sqrt(stand_err_diff + stand_err_diff.T)
        stand_err_diff[stand_err_diff == 0.0] = 1e-8
        z_pairs = mos_pairs / stand_err_diff

        # % Include everything above the diagonal.
        # delta_vqm = [];
        # z = [];
        # for col = 2:num_comb
        #     delta_vqm = [delta_vqm; vqm_pairs(1:col-1,col)];
        #     z = [z; z_pairs(1:col-1,col)];
        # end
        delta_vqm = []
        z = []
        for col in range(2, num_comb + 1):
            delta_vqm = np.hstack([delta_vqm, vqm_pairs[0:col-1, col-1]])
            z = np.hstack([z, z_pairs[0:col-1, col-1]])

        # % Switch on z and delta_vqm for negative delta_vqm
        # z_vqm = z;
        # negs_vqm = find(delta_vqm < 0);
        # delta_vqm(negs_vqm) = -delta_vqm(negs_vqm);
        # z_vqm(negs_vqm) = -z_vqm(negs_vqm);
        z_vqm = z
        negs_vqm = indices(delta_vqm, lambda x: x < 0)
        delta_vqm[negs_vqm] = - delta_vqm[negs_vqm]
        z_vqm[negs_vqm] = - z_vqm[negs_vqm]

        # % Compute the average confidence that vqm(2) is worse than vqm(1) in mean_cdf_z_vqm.
        # cdf_z_vqm = .5+erf(z_vqm/sqrt(2))/2;
        cdf_z_vqm = .5 + scipy.special.erf(z_vqm/np.sqrt(2))/2

        # === original binning logic: ===
        # % One control parameter for delta_vqm resolution plot; number of vqm bins,
        # % equally spaced from min(delta_vqm) to max(delta_vqm).

        # % Sliding neighborhood filter with 50% overlap means that there will actually
        # % be vqm_bins*2-1 points on the delta_vqm resolution plot.
        # vqm_bins = 10; % How many bins to divide full vqm range for local averaging
        # vqm_low = min(delta_vqm); % lower limit on delta_vqm
        # vqm_high = max(delta_vqm); % upper limit on delta_vqm
        # vqm_step = (vqm_high-vqm_low)/vqm_bins; % size of delta_vqm bins
        vqm_bins = 10
        vqm_low = min(delta_vqm)
        vqm_high = max(delta_vqm)
        vqm_step = (vqm_high - vqm_low) / vqm_bins

        # % lower, upper, and center bin locations
        # low_limits = [vqm_low:vqm_step/2:vqm_high-vqm_step];
        # high_limits = [vqm_low+vqm_step:vqm_step/2:vqm_high];
        # centers = [vqm_low+vqm_step/2:vqm_step/2:vqm_high-vqm_step/2];
        low_limits = np.arange(vqm_low, vqm_high - vqm_step, step=vqm_step / 2)
        centers = low_limits.copy() + vqm_step / 2
        high_limits = low_limits.copy() + vqm_step
        # patch to cover entire range
        if high_limits[-1] < vqm_high:
            low_limits = np.hstack([low_limits, vqm_high - vqm_step])
            high_limits = np.hstack([high_limits, vqm_high])
            centers = np.hstack([centers, vqm_high - vqm_step / 2])

        len_centers = len(centers)
        assert len_centers == len(low_limits) == len(high_limits)

        # mean_cdf_z_vqm = zeros(1,2*vqm_bins-1);
        # for i=1:2*vqm_bins-1
        #     in_bin = find(low_limits(i) <= delta_vqm & delta_vqm < high_limits(i));
        #     mean_cdf_z_vqm(i) = mean(cdf_z_vqm(in_bin));
        # end
        mean_cdf_z_vqm = np.zeros(len_centers)
        for i in range(0, len_centers):
            in_bin = indices(delta_vqm, lambda x: low_limits[i] <= x < high_limits[i])
            if len(in_bin) == 0:
                mean_cdf_z_vqm[i] = float('NaN')
            else:
                mean_cdf_z_vqm[i] = np.mean(cdf_z_vqm[in_bin])
        centers__mean_cdf_z_vqm = filter(lambda p: not np.isnan(p[1]), zip(centers, mean_cdf_z_vqm))
        centers, mean_cdf_z_vqm = zip(*centers__mean_cdf_z_vqm)

        # # % % Optional code to plot resolving power curve.
        # # % % The x-axis is vqm(2)-vqm(1).  The Y-axis is always the average
        # # % % confidence that vqm(2) is worse than vqm(1).
        # # % figure(1)
        # # % plot(centers,mean_cdf_z_vqm)
        # # % grid
        # # % set(gca,'LineWidth',1)
        # #
        # # % set(gca,'FontName','Ariel')
        # # % set(gca,'fontsize',11)
        # # % xlabel('VQM (2) - VQM (1)')
        # # % ylabel('Average Confidence VQM (2) is worse than VQM (1)')
        # # % title('VQM Resolving Power')
        #
        # # % Compute each resolving power by interpolating the mean_cdf_z_vqm graph
        #
        # # % 95% resolving power
        # # i = length(centers) - 1;
        # # while mean_cdf_z_vqm(i) > 0.95 && i > 1,
        # #     i = i -1;
        # # end
        # # j = min(length(centers), i+1);
        # # resolving_power(1) = interp1(mean_cdf_z_vqm(i:j),centers(i:j), 0.95);
        #
        # # % 90% resolving power
        # # i = length(centers) - 1;
        # # while mean_cdf_z_vqm(i) > 0.90 && i > 1,
        # # i = i -1;
        # # end
        # # j = min(length(centers), i+1);
        # # resolving_power(2) = interp1(mean_cdf_z_vqm(i:j),centers(i:j), 0.90);
        #
        # # % 75% resolving power
        # # i = length(centers) - 1;
        # # while mean_cdf_z_vqm(i) > 0.75 && i > 1,
        # # i = i -1;
        # # end
        # # j = min(length(centers), i+1);
        # # resolving_power(3) = interp1(mean_cdf_z_vqm(i:j),centers(i:j), 0.75);
        #
        # # % 68% resolving power
        # # i = length(centers) - 1;
        # # while mean_cdf_z_vqm(i) > 0.68 && i > 1,
        # # i = i -1;
        # # end
        # # j = min(length(centers), i+1);
        # # resolving_power(4) = interp1(mean_cdf_z_vqm(i:j),centers(i:j), 0.68);
        #
        # resolving_powers = []
        # for perc in [0.95, 0.90, 0.75, 0.68]:
        #     i = len(centers) - 1
        #     while mean_cdf_z_vqm[i-1] > perc and i > 1:
        #         i -= 1
        #     j = min(len(centers), i+1)
        #     resolving_power = scipy.interpolate.interp1d(mean_cdf_z_vqm[i-1:j], centers[i-1:j])(perc)
        #     resolving_powers.append(resolving_power)

        try:
            res_pow_95 = scipy.interpolate.interp1d(mean_cdf_z_vqm, centers, kind='linear')([0.95])[0]
        except ValueError:
            res_pow_95 = float('NaN')

        # % return infinity if can't compute
        # resolving_power(isnan(resolving_power)) = inf;

        result = dict()
        result['resolving_power_95perc'] = res_pow_95
        result['score'] = res_pow_95
        return result


class AggrScorePerfMetric(PerfMetric):
    """
    Groundtruth is a list of aggregate scores (list of real numbers)
    """

    @staticmethod
    def sigmoid_adjust(xs, ys):
        ys_max = np.max(ys) + 0.1
        ys_min = np.min(ys) - 0.1

        # normalize to [0, 1]
        ys = list((np.array(ys) - ys_min) / (ys_max - ys_min))

        zs = -np.log(1.0 / np.array(ys).T - 1.0)
        Y_mtx = np.array((np.ones(len(ys)), zs)).T
        x_vec = np.array([xs]).T
        a_b = lstsq(Y_mtx, x_vec, rcond=-1)[0]
        a = a_b.item(0)
        b = a_b.item(1)

        xs = 1.0 / (1.0 + np.exp(- (np.array(xs) - a) / b))

        # denormalize
        xs = xs * (ys_max - ys_min) + ys_min

        return xs

    @classmethod
    def _preprocess(cls, groundtruths, predictions, **kwargs):
        aggre_method = kwargs['aggr_method'] if 'aggr_method' in kwargs else np.mean
        enable_mapping = kwargs['enable_mapping'] if 'enable_mapping' in kwargs else False

        groundtruths_ = list(map(
            lambda x: aggre_method(x) if hasattr(x, '__len__') else x,
            groundtruths))

        if enable_mapping:
            predictions_ = cls.sigmoid_adjust(predictions, groundtruths_)
        else:
            predictions_ = predictions

        return groundtruths_, predictions_


class RmsePerfMetric(AggrScorePerfMetric):

    TYPE = "RMSE"
    VERSION = "1.0"

    @classmethod
    def _evaluate(cls, groundtruths, predictions, **kwargs):
        rmse = np.sqrt(np.mean(np.power(np.array(groundtruths) - np.array(predictions), 2.0)))
        result = {'score': rmse}
        return result


class SrccPerfMetric(AggrScorePerfMetric):

    TYPE = "SRCC"
    VERSION = "1.0"

    @classmethod
    def _evaluate(cls, groundtruths, predictions, **kwargs):
        # spearman
        srcc, _ = scipy.stats.spearmanr(groundtruths, predictions)
        result = {'score': srcc}
        return result


class PccPerfMetric(AggrScorePerfMetric):

    TYPE = "PCC"
    VERSION = "1.0"

    @classmethod
    def _evaluate(cls, groundtruths, predictions, **kwargs):
        # pearson
        pcc, _ = scipy.stats.pearsonr(groundtruths, predictions)
        result = {'score': pcc}
        return result


class KendallPerfMetric(AggrScorePerfMetric):

    TYPE = "KENDALL"
    VERSION = "1.0"

    @classmethod
    def _evaluate(cls, groundtruths, predictions, **kwargs):
        # kendall
        kendall, _ = scipy.stats.kendalltau(groundtruths, predictions)
        result = {'score': kendall}
        return result
