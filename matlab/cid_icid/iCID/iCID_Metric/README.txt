This is a MATLAB implementation of the improved color-image-difference metric (iCID metric) which predicts the perceived difference of two images.

The code is written for MATLAB R2010b and might not work properly with different MATLAB versions.

It is supplementary material to the following paper:

J. Preiss, F. Fernandes, and P. Urban, "Color-Image Quality Assessment: From Prediction to Optimization", IEEE Transactions on Image Processing, pp. 1366-1378, Volume 23, Issue 3, March 2014
For questions, please contact the authors: preiss.science@gmail.com, fernandes@idd.tu-darmstadt.de,  philipp.urban@igd.fraunhofer.de

The implementation of the iCID metric is available:
- 'iCID.m'

An example of use is also available:
- 'Example.m'.


VERSION HISTORY
---------------
2013/04/03: Version 0.90
2013/05/24: Version 0.91
2014/02/12: Version 0.92
2014/02/27: Version 1.00


ACKNOWLEDGMENT DFG
------------------
This work was supported by the German Research Foundation (DFG).


ACKNOWLEDGMENT EXAMPLE IMAGE
----------------------------
The example image 'ExampleImages/Image1.tif' (with a gamut-mapped version 'ExampleImages/Image2.tif') is taken from Fotopedia:
Chris Willis, “Standford holi festival : pretty girl”, Jan 2010, [Accessed: Apr-03-2013]. [Online]. Available: http://www.fotopedia.com/items/flickr-4474731834.


ACKNOWLEDGMENT 'COLORSPACE' PACKAGE
-----------------------------------
This code uses the 'colorspace' package by Pascal Getreuer (http://www.getreuer.info/home/colorspace) under a BSD license:

Copyright © 2005–2010, Pascal Getreuer
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.