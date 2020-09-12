This is a MATLAB implementation of the gamut-mapping optimization which optimizes a gamut-mapped image by minimizing the iCID metric of the original image and the gamut-mapped image.

The 64 bit MATLAB exchange function from C++ source code (.mexw64) was compiled by Microsoft Visual Studio 2008 using the Microsoft Visual C++ 2005 Redistributable (x64) Version 8.0.61000. The gamut-mapping optimization might not work properly on a 32 bit system.

The MATLAB code is written for MATLAB R2010b and might not work properly with different MATLAB versions.

It is supplementary material to the following paper:

J. Preiss, F. Fernandes, and P. Urban, "Color-Image Quality Assessment: From Prediction to Optimization", IEEE Transactions on Image Processing, pp. 1366-1378, Volume 23, Issue 3, March 2014
For questions, please contact the authors: preiss.science@gmail.com, fernandes@idd.tu-darmstadt.de,  philipp.urban@igd.fraunhofer.de

The implementation of the gamut-mapping optimization is available:
- 'Optimizing_Gamut_Mapping.m'

An example of use is also available:
- 'Example.m'.


VERSION HISTORY
---------------
2013/05/24: Version 0.91
2014/02/12: Version 0.92
2014/02/27: Version 1.00


ACKNOWLEDGMENT DFG
------------------
This work was supported by the German Research Foundation (DFG).


ACKNOWLEDGMENT EXAMPLE IMAGE
----------------------------
The example image 'ExampleImages/OriginalImg.tif' (with a gamut-mapped version 'ExampleImages/MappedImg.tif') is taken from Wimimedia Commons:
Friedrich Boehringer, “Tautropfen auf Gras”, Aug 2009, [Accessed: May-24-2013]. [Online]. Available: http://commons.wikimedia.org/wiki/File:Tropfen_auf_Gras_2.JPG


ACKNOWLEDGMENT 'COLORSPACE' PACKAGE
-----------------------------------
This code uses the 'colorspace' package by Pascal Getreuer (http://www.getreuer.info/home/colorspace) under a BSD license:

Copyright © 2005–2010, Pascal Getreuer
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.