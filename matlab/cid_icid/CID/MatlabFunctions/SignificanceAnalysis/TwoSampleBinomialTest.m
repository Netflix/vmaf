% Performs the "Two Sample Binomial Test" according to Yule. See L. Brown
% and X. Li, "Confidence intervals for two sample binomial distribution",
% Journal of Statistical Planning and Inference, Vol. 130, pp. 359–375,
% 2005.
%
% This code is supplementary material to the article:
%           I. Lissner, J. Preiss, P. Urban, M. Scheller Lichtenauer, and 
%           P. Zolliker, "Image-Difference Prediction: From Grayscale to
%           Color", IEEE Transactions on Image Processing (accepted), 2012.
%
% Authors:  Ingmar Lissner, Jens Preiss, Philipp Urban
%           Institute of Printing Science and Technology
%           Technische Universität Darmstadt
%           {lissner,preiss,urban}@idd.tu-darmstadt.de
%           http://www.idd.tu-darmstadt.de/color
%
%           Matthias Scheller Lichtenauer, Peter Zolliker
%           Empa, Swiss Federal Laboratories for
%                 Materials Science and Technology
%           Laboratory for Media Technology
%           {matthias.scheller,peter.zolliker}@empa.ch
%           http://empamedia.ethz.ch
%
% Interface:
%           [h, Interval] = TwoSampleBinomialTest(X, Y, m, alpha)
%
% Parameters:
% h         Result of the test: h = 1 -> success probabilities are
%                                        most likely different
% Interval  Confidence interval of |p1-p2|
%
% X         Number of successful trials in sample X
% Y         Number of successful trials in sample Y
% m         Number of trials
% alpha     Confidence level; typically, alpha = 0.05
function [h, Interval] = TwoSampleBinomialTest(X, Y, m, alpha)
p1 = X/m;
p2 = Y/m;
pp = (X+Y)/(2*m);                                   % Brown: p_dash
qq = 1-pp;                                          % Brown: q_dash
quantile = norminv(1-alpha/2);                      % Brown: z_(alpha/2)
Interval = abs(p1-p2) - quantile*sqrt(2/m*pp*qq);	% Brown: Eq. (2)
% Interval = [Interval, abs(p1-p2) + quantile*sqrt(2/m*pp*qq)];
h = Interval(1,1) > 0;

% Explanation: We take the absolute value of (p1-p2) to make this function
% symmetric (independent of the order of the input arguments X and Y). The
% probabilities are assumed to be significantly different if the confidence
% interval does not contain 0. In our case, the interval can either
%   a) contain 0
%   b) be above zero, i.e., lower and upper bound > 0
% We check if b) is fulfilled by testing if the lower bound is > 0. This is
% sufficient, as the upper bound is always greater than the lower bound.