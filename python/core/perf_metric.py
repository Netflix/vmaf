import numpy as np
from numpy.linalg import lstsq
import scipy.stats

from core.mixin import TypeVersionEnabled
from tools.misc import empty_object, indices
from tools.sigproc import fastDeLong, calpvalue, significanceHM, \
    significanceBinomial

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

class PerfMetric(TypeVersionEnabled):

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
        assert len(self.groundtruths) == len(self.predictions)

    def evaluate(self, **kwargs):
        """
        :return: ret - a dictionary with 'score' and other keys
        """
        groundtruths, predictions = self._preprocess(self.groundtruths, self.predictions, **kwargs)
        result = self._evaluate(groundtruths, predictions, **kwargs)
        assert 'score' in result
        return result

class RawScorePerfMetric(PerfMetric):
    """
    Groundtruth is a list of raw scores (list of list of real numbers)
    """
    def _assert_args(self):
        # override PerfMetric._assert_args
        super(RawScorePerfMetric, self)._assert_args()

        # require the raw scores to be more than 1
        for groundtruth in self.groundtruths:
            assert hasattr(groundtruth, '__len__') and len(groundtruth) > 1

class KflkPerfMetric(RawScorePerfMetric):
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

    TYPE = "KFLK"
    VERSION = "0.1"

    @classmethod
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
        D = np.abs(objScoDif[:, indices(signif[0], lambda x: x!=0)])
        S = np.abs(objScoDif[:, indices(signif[0], lambda x: x==0)])
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
            for j in range(i+1, M+1):
                # http://stackoverflow.com/questions/4257394/slicing-of-a-numpy-2d-array-or-how-do-i-extract-an-mxm-submatrix-from-an-nxn-ar
                pDS_DL[i-1, j-1] = calpvalue(AUC_DS[[i-1, j-1]], C[[[i-1],[j-1]],[i-1, j-1]])
                pDS_DL[j-1, i-1] = pDS_DL[i-1, j-1]

        # [pDS_HM,CI_DS] = significanceHM(S, D, AUC_DS);
        pDS_HM, CI_DS = significanceHM(S, D, AUC_DS)

        # THR = prctile(D',95);
        THR = np.percentile(D, 95, axis=1)

        # %%%%%%%%%%%%%%%%%%%%%%% Better / Worse %%%%%%%%%%%%%%%%%%%%%%%%%%%

        # B = [objScoDif(:,signif == 1),-objScoDif(:,signif == -1)];
        # W = -B;
        # samples.ratings = [B,W];
        # samples.spsizes = [size(B,2),size(W,2)];
        B1 = objScoDif[:, indices(signif[0], lambda x: x== 1)]
        B2 = objScoDif[:, indices(signif[0], lambda x: x==-1)]
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
            CC_0[m] = float(np.sum(B[m,:]>0) + np.sum(W[m,:]<0)) / L

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
            for j in range(i+1, M+1):
                pBW_DL[i-1, j-1] = calpvalue(AUC_BW[[i-1, j-1]], C[[[i-1],[j-1]],[i-1, j-1]])
                pBW_DL[j-1, i-1] = pBW_DL[i-1, j-1]

                pCC0_b[i-1, j-1] = significanceBinomial(CC_0[i-1], CC_0[j-1], L)
                pCC0_b[j-1, i-1] = pCC0_b[i-1, j-1]

                # pCC0_F[i-1, j-1] = fexact(CC_0[i-1]*L, 2*L, CC_0[i-1]*L + CC_0[j-1]*L, L, 'tail', 'b') / 2.0
                # pCC0_F[j-1, i-1] = pCC0_F[i-1,j]

        # [pBW_HM,CI_BW] = significanceHM(B, W, AUC_BW);
        pBW_HM,CI_BW = significanceHM(B, W, AUC_BW)

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
            'pDS_HM': pDS_HM,
            'AUC_BW': AUC_BW,
            'pBW_DL': pBW_DL,
            'pBW_HM': pBW_HM,
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

        def _signif(a, b):
            mos_a = np.mean(a)
            mos_b = np.mean(b)
            n_a = len(a)
            n_b = len(b)
            var_a = np.var(a, ddof=1)
            var_b = np.var(b, ddof=1)
            z = (mos_a - mos_b) / np.sqrt(var_a/n_a + var_b/n_b)
            if z < -2:
                return -1
            elif z> 2:
                return 1
            else:
                return 0

        # generate pairs
        N = len(groundtruths)
        objscodif_mtx = np.zeros([N, N])
        signif_mtx = np.zeros([N, N])
        i = 0
        for groundtruth, prediction in zip(groundtruths, predictions):
            j = 0
            for groundtruth2, prediction2 in zip(groundtruths, predictions):
                objscodif = prediction - prediction2
                signif = _signif(groundtruth, groundtruth2)
                objscodif_mtx[i, j] = objscodif
                signif_mtx[i, j] = signif
                j += 1
            i += 1

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(objscodif_mtx, interpolation='nearest')
        # plt.set_cmap('gray')
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(signif_mtx, interpolation='nearest')
        # plt.set_cmap('gray')
        # plt.colorbar()
        # plt.show()

        results = cls._metrics_performance(objscodif_mtx.reshape(1, N*N), signif_mtx.reshape(1, N*N))

        # _metrics_performance allows processing multiple objective quality
        # metrics together. Here we just process one:
        result = {}
        for key in results:
            result[key] = results[key][0]

        result['score'] = result['AUC_DS']

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
        Y_mtx = np.matrix((np.ones(len(ys)), zs)).T
        x_vec = np.matrix(xs).T
        a_b = lstsq(Y_mtx, x_vec)[0]
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

        groundtruths_ = map(
            lambda x: aggre_method(x) if hasattr(x, '__len__') else x,
            groundtruths)

        if enable_mapping:
            predictions_ = cls.sigmoid_adjust(predictions, groundtruths_)
        else:
            predictions_ = predictions

        return groundtruths_, predictions_

class RmsePerfMetric(AggrScorePerfMetric):

    TYPE = "RMSE"
    VERSION = "1.0"

    @staticmethod
    def _evaluate(groundtruths, predictions, **kwargs):
        rmse = np.sqrt(np.mean(np.power(np.array(groundtruths) - np.array(predictions), 2.0)))
        result = {'score': rmse}
        return result

class SrccPerfMetric(AggrScorePerfMetric):

    TYPE = "SRCC"
    VERSION = "1.0"

    @staticmethod
    def _evaluate(groundtruths, predictions, **kwargs):
        # spearman
        srcc, _ = scipy.stats.spearmanr(groundtruths, predictions)
        result = {'score': srcc}
        return result

class PccPerfMetric(AggrScorePerfMetric):

    TYPE = "PCC"
    VERSION = "1.0"

    @staticmethod
    def _evaluate(groundtruths, predictions, **kwargs):
        # pearson
        pcc, _ = scipy.stats.pearsonr(groundtruths, predictions)
        result = {'score': pcc}
        return result

class KendallPerfMetric(AggrScorePerfMetric):

    TYPE = "KENDALL"
    VERSION = "1.0"

    @staticmethod
    def _evaluate(groundtruths, predictions, **kwargs):
        # kendall
        kendall, _ = scipy.stats.kendalltau(groundtruths, predictions)
        result = {'score': kendall}
        return result
