import numpy as np
import scipy.misc
import scipy.ndimage
import scipy.stats
import scipy.io
from PIL import Image

from vmaf.config import VmafConfig
from vmaf.tools.misc import index_and_value_of_min

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


def _gauss_window(lw, sigma):
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


def _hp_image(image):
    extend_mode = 'reflect'
    image = np.array(image).astype(np.float32)
    w, h = image.shape
    mu_image = np.zeros((w, h))
    _avg_window = _gauss_window(3, 1.0)
    scipy.ndimage.correlate1d(image, _avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, _avg_window, 1, mu_image, mode=extend_mode)
    return image - mu_image


def _var_image(hpimg):
    extend_mode = 'reflect'
    w, h = hpimg.shape
    varimg = np.zeros((w, h))
    _var_window = _gauss_window(3, 1.0)
    scipy.ndimage.correlate1d(hpimg**2, _var_window, 0, varimg, mode=extend_mode)
    scipy.ndimage.correlate1d(varimg, _var_window, 1, varimg, mode=extend_mode)
    return varimg


def as_one_hot(label_list):
    return np.eye(2)[np.array(label_list).astype(np.int)]


def create_hp_yuv_4channel(yuvimg):
    yuvimg = yuvimg.astype(np.float32)
    yuvimg /= 255.0
    hp_y = _hp_image(yuvimg[:, :, 0])
    hp_u = _hp_image(yuvimg[:, :, 1])
    hp_v = _hp_image(yuvimg[:, :, 2])
    sigma = np.sqrt(_var_image(hp_y))

    # stack together to make 4 channel image
    return np.dstack((hp_y, hp_u, hp_v, sigma))


def dstack_y_u_v(y, u, v):
    # make y, u, v consistent in size
    if u.shape != y.shape:
        u = np.array(Image.fromarray(u).convert("L").resize(y.shape, Image.BICUBIC))
    if v.shape != y.shape:
        v = np.array(Image.fromarray(v).convert("L").resize(y.shape, Image.BICUBIC))
    return np.dstack((y, u, v))


def midrank(x):
    # [Z J]=sort(x);
    # Z=[Z Z(end)+1];
    # N=length(x);
    # T=zeros(1,N);
    J, Z = zip(*sorted(enumerate(x), key=lambda x:x[1]))
    J = list(J)
    Z = list(Z)
    Z.append(Z[-1]+1)
    N = len(x)
    T = np.zeros(N)

    # i=1;
    # while i<=N
    #     a=i;
    #     j=a;
    #     while Z(j)==Z(a)
    #         j=j+1;
    #     end
    #         b=j-1;
    #     for k=a:b
    #         T(k)=(a+b)/2;
    #     end
    #     i=b+1;
    # end
    i = 1
    while i <= N:
        a = i
        j = a
        while Z[j-1] == Z[a-1]:
            j = j + 1
        b = j - 1
        for k in range(a, b+1):
            T[k-1] = (a + b) / 2
        i = b + 1

    # T(J)=T;
    T2 = np.zeros(N)
    T2[J] = T

    return T2


def calpvalue(aucs, sigma):
    # function pvalue = calpvalue(aucs, sigma)
    # l = [1, -1];
    # z = abs(diff(aucs)) / sqrt(l * sigma * l');
    # pvalue = 2 * (1 - normcdf(z, 0, 1));
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    pvalue = 2 * (1 - scipy.stats.norm.cdf(z, loc=0, scale=1))
    return pvalue


def _cov_kendall(x):
    """
    x: rows - observation vector 0, 1, 2, ...
    return a covariance matrix based on kendall correlation
    """
    m, n = x.shape
    cov_ = np.zeros([m, m])
    for i in range(m):
        for j in range(i, m):
            # scipy 1.2.0 kendalltau() has an issue with the p-value of two long vectors that are perfectly monotonic
            # see here: https://github.com/scipy/scipy/issues/9611
            # until this is fixed, we bypass the exact calculation method by using the variance approximation (asymptotic method)
            # need a try-except clause: ealier scipy versions do not support a method keywarg
            try:
                kendall, _ = scipy.stats.kendalltau(x[i,:], x[j,:], method='asymptotic')
            except TypeError:
                kendall, _ = scipy.stats.kendalltau(x[i, :], x[j, :])
            cov_[i, j] = kendall
            cov_[j, i] = kendall
    return cov_


def AUC_CI(n_D, n_I, Area):
    # function [CI,SE] = AUC_CI(n_D,n_I,Area)
    #
    # % By Lukas Krasula
    # % Inspired by
    # % *********************  CIAUC  ****************************
    # %   (c) John W Pickering, Novemeber 2009
    # %     Christchurch Kidney Research Group
    # %     University of Otago Christchurch
    # %     New Zealand
    # %
    # %   Last update:  17 July 2012
    # %
    # %	Redistribution and use in source and binary forms, with or without
    # %   modification, are permitted provided that the following conditions are met:
    # %
    # %   * Redistributions of source code must retain the above copyright
    # %     notice, this list of conditions and the following disclaimer.
    # %   * Redistributions in binary form must reproduce the above copyright
    # %     notice, this list of conditions and the following disclaimer in
    # %     the documentation and/or other materials provided with the distribution
    # %
    # % Attribution to John Pickering.
    # % *************************************************************************
    # % n_D - number of different pairs
    # % n_I - number of indifferent pairs
    # % Area - Area under ROC curve

    # Q1=Area/(2-Area);
    # Q2=2*Area*Area/(1+Area);
    Q1 = Area / (2.0 - Area)
    Q2 = 2.0 * Area * Area / (1.0 + Area)

    # SE=sqrt((Area*(1-Area)+(n_D-1)*(Q1-Area*Area)+(n_I-1)*(Q2-Area*Area))/(n_I*n_D));
    SE = np.sqrt((Area * (1.0 - Area) + (n_D-1) * (Q1 - Area * Area) +
                  (n_I - 1.0) * (Q2 - Area*Area)) / (n_I * n_D))

    # CI = 1.96 * SE;
    CI = 1.96 * SE

    return CI, SE


def significanceHM(A, B, AUCs):
    # function [pHM,CI] = significanceHM(A,B,AUCs)
    # % By Lukas Krasula

    assert A.shape[0] == B.shape[0] == AUCs.shape[0]

    # n_met = size(A,1);
    n_met = A.shape[0]

    # CorrA = corr(A','type','Kendall');
    # CorrB = corr(B','type','Kendall');
    CorrA = _cov_kendall(A)
    CorrB = _cov_kendall(B)

    # pHM = ones(n_met);
    # CI = ones(n_met,1);
    # for i=1:n_met-1
    #
    #     [CI(i),SE1] = AUC_CI(size(A,2),size(B,2),AUCs(i));
    #
    #     for j=i+1:n_met
    #         [CI(j),SE2] = AUC_CI(size(A,2),size(B,2),AUCs(j));
    #
    #         load('Hanley_McNeil.mat');
    #
    #         rA = (CorrA(i,j) + CorrB(i,j))/2;
    #         AA = (AUCs(i) + AUCs(j))/2;
    #
    #         [~,rr] = min(abs(rA-rA_vec));
    #         [~,aa] = min(abs(AA-AA_vec));
    #         r = Table_HM(rr,aa);
    #
    #         z = abs(AUCs(i) - AUCs(j)) / sqrt( SE1^2 + SE2^2 + 2*r*SE1*SE2 );
    #         pHM(i,j) = 1-normcdf(z);
    #         pHM(j,i) = pHM(i,j);
    #     end
    # end
    hm_filepath = VmafConfig.tools_resource_path('Hanley_McNeil.mat')
    hm_dict = scipy.io.loadmat(hm_filepath)
    pHM = np.ones([n_met, n_met])
    Table_HM = hm_dict['Table_HM']
    AA_vec = hm_dict['AA_vec']
    rA_vec = hm_dict['rA_vec']
    CI = np.ones(n_met)
    for i in range(1, n_met):
        CI1,SE1 = AUC_CI(A.shape[1], B.shape[1], AUCs[i-1])
        CI[i-1] = CI1

        for j in range(i+1, n_met+1):
            CI2, SE2 = AUC_CI(A.shape[1], B.shape[1], AUCs[j-1])
            CI[j-1] = CI2

            rA = (CorrA[i-1,j-1] + CorrB[i-1,j-1]) / 2
            AA = (AUCs[i-1] + AUCs[j-1]) / 2

            rr, _ = index_and_value_of_min(abs(rA - rA_vec).ravel())
            aa, _ = index_and_value_of_min(abs(AA - AA_vec).ravel())
            r = Table_HM[rr, aa]

            z = abs(AUCs[i - 1] - AUCs[j - 1]) / np.sqrt(SE1 ** 2 + SE2 ** 2 + 2 * r * SE1 * SE2)
            pHM[i-1, j-1] = 1.0 - scipy.stats.norm.cdf(z)
            pHM[j-1, i-1] = pHM[i-1, j-1]

    return pHM, CI


def fastDeLong(samples):
    # %FASTDELONGCOV
    # %The fast version of DeLong's method for computing the covariance of
    # %unadjusted AUC.
    # %% Reference:
    # % @article{sun2014fast,
    # %   title={Fast Implementation of DeLong's Algorithm for Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
    # %   author={Xu Sun and Weichao Xu},
    # %   journal={IEEE Signal Processing Letters},
    # %   volume={21},
    # %   number={11},
    # %   pages={1389--1393},
    # %   year={2014},
    # %   publisher={IEEE}
    # % }
    # %% [aucs, delongcov] = fastDeLong(samples)
    # %%
    # % Edited by Xu Sun.
    # % Homepage: https://pamixsun.github.io
    # % Version: 2014/12
    # %%

    # if sum(samples.spsizes) ~= size(samples.ratings, 2) || numel(samples.spsizes) ~= 2
    #     error('Argument mismatch error');
    # end
    if np.sum(samples.spsizes) != samples.ratings.shape[1] or len(samples.spsizes) != 2:
        assert False, 'Argument mismatch error'

    # z = samples.ratings;
    # m = samples.spsizes(1);
    # n = samples.spsizes(2);
    # x = z(:, 1 : m);
    # y = z(:, m + 1 : end);
    # k = size(z, 1);
    z = samples.ratings
    m, n = samples.spsizes
    x = z[:, :m]
    y = z[:, m:]
    k = z.shape[0]

    # tx = zeros(k, m);
    # ty = zeros(k, n);
    # tz = zeros(k, m + n);
    # for r = 1 : k
    #     tx(r, :) = midrank(x(r, :));
    #     ty(r, :) = midrank(y(r, :));
    #     tz(r, :) = midrank(z(r, :));
    # end
    tx = np.zeros([k, m])
    ty = np.zeros([k, n])
    tz = np.zeros([k, m + n])
    for r in range(k):
        tx[r, :] = midrank(x[r, :])
        ty[r, :] = midrank(y[r, :])
        tz[r, :] = midrank(z[r, :])

    # % tz
    # aucs = sum(tz(:, 1 : m), 2) / m / n - (m + 1) / 2 / n;
    # v01 = (tz(:, 1 : m) - tx(:, :)) / n;
    # v10 = 1 - (tz(:, m + 1 : end) - ty(:, :)) / m;
    # sx = cov(v01')';
    # sy = cov(v10')';
    # delongcov = sx / m + sy / n;
    aucs = np.sum(tz[:, :m], axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n

    return aucs, delongcov, v01, v10


def significanceBinomial(p1, p2, N):
    # function pValue = significanceBinomial(p1,p2,N)
    # p = (p1+p2) / 2;
    # sigmaP1P2 = sqrt(p*(1-p)*2/N);
    # z = abs(p1-p2)/sigmaP1P2;
    # pValue = 2*(1 - normcdf(z, 0, 1));
    p = (p1 + p2) / 2.0
    sigmaP1P2 = np.sqrt(p * (1.0 - p) * 2.0 / N)
    z = abs(p1 - p2) / sigmaP1P2
    pValue = 2.0 * (1.0 - scipy.stats.norm.cdf(z, 0.0, 1.0))

    return pValue
