This is a MATLAB implementation of the color-image-difference measure (CID measure), which predicts the perceived difference of two images.

It is supplementary material to the following paper:

I. Lissner, J. Preiss, P. Urban, M. Scheller Lichtenauer, and P. Zolliker, "Image-Difference Prediction: From Grayscale to Color", IEEE Transactions on Image Processing, Vol. 22, Issue 2, pp. 435-446 (2013).

Two implementations of the CID measure are available:
- 'CID.m'			Easy-to-use example with default parameters
- 'CID_advanced.m'		Allows to set the parameters manually

An example of use is provided in 'Example.m'.

An implementation of the significance analysis used in the above paper is also included: 'MatlabFunctions/SignificanceAnalysis/TwoSampleBinomialTest.m'.


VERSION HISTORY
---------------
2013/05/31: Version 1.0, paper published
2012/08/24: Version 0.91, sL12 now computed without abs (as in the paper)
2012/08/14: Version 0.9


ACKNOWLEDGMENT DFG
------------------
This work was supported by the German Research Foundation (DFG).


ACKNOWLEDGMENT 'PARROTS'
------------------------
The images in 'ExampleImages' are part of the "Kodak Lossless True Color Image Suite" (http://r0k.us/graphics/kodak/).


ACKNOWLEDGMENT 'COLORSPACE' PACKAGE
-----------------------------------
This code uses the 'colorspace' package by Pascal Getreuer (http://www.getreuer.info/home/colorspace) under a BSD license:

Copyright © 2005–2010, Pascal Getreuer
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


ACKNOWLEDGMENT 'S-CIELAB' IMPLEMENTATION
----------------------------------------
This code uses the 'S-CIELAB' impementation bei Xuemei Zhang (http://white.stanford.edu/~brian/scielab/).