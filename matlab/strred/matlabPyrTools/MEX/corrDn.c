/* 
RES = corrDn(IM, FILT, EDGES, STEP, START, STOP);
  >>> See corrDn.m for documentation <<<
  This is a matlab interface to the internal_reduce function. 
  EPS, 7/96.
*/

#define V4_COMPAT
#include <matrix.h>  /* Matlab matrices */
#include <mex.h>

#include "convolve.h"

#define notDblMtx(it) (!mxIsNumeric(it) || !mxIsDouble(it) || mxIsSparse(it) || mxIsComplex(it))

void mexFunction(int nlhs,	     /* Num return vals on lhs */
		 mxArray *plhs[],    /* Matrices on lhs      */
		 int nrhs,	     /* Num args on rhs    */
		 const mxArray *prhs[]     /* Matrices on rhs */
		 )
  {
  double *image,*filt, *temp, *result;
  int x_fdim, y_fdim, x_idim, y_idim;
  int x_rdim, y_rdim;
  int x_start = 1;
  int x_step = 1;
  int y_start = 1;
  int y_step = 1;
  int x_stop, y_stop;
  mxArray *arg;
  double *mxMat;
  char edges[15] = "reflect1";
  
  if (nrhs<2) mexErrMsgTxt("requres at least 2 args.");

  /* ARG 1: IMAGE  */
  arg = prhs[0];
  if notDblMtx(arg) mexErrMsgTxt("IMAGE arg must be a non-sparse double float matrix.");
  image = mxGetPr(arg);
  x_idim = (int) mxGetM(arg); /* X is inner index! */
  y_idim = (int) mxGetN(arg);

  /* ARG 2: FILTER */
  arg = prhs[1];
  if notDblMtx(arg) mexErrMsgTxt("FILTER arg must be non-sparse double float matrix.");
  filt = mxGetPr(arg);
  x_fdim = (int) mxGetM(arg); 
  y_fdim = (int) mxGetN(arg);

  if ((x_fdim > x_idim) || (y_fdim > y_idim))
    {
    mexPrintf("Filter: [%d %d], Image: [%d %d]\n",x_fdim,y_fdim,x_idim,y_idim);
    mexErrMsgTxt("FILTER dimensions larger than IMAGE dimensions.");
    }

  /* ARG 3 (optional): EDGES */
  if (nrhs>2) 
      {
      if (!mxIsChar(prhs[2]))
	mexErrMsgTxt("EDGES arg must be a string.");
      mxGetString(prhs[2],edges,15);
      }

  /* ARG 4 (optional): STEP */
  if (nrhs>3)
      {
      arg = prhs[3];
      if notDblMtx(arg) mexErrMsgTxt("STEP arg must be a double float matrix.");
      if (mxGetM(arg) * mxGetN(arg) != 2)
    	 mexErrMsgTxt("STEP arg must contain two elements.");
      mxMat = mxGetPr(arg);
      x_step = (int) mxMat[0];
      y_step = (int) mxMat[1];
      if ((x_step<1) || (y_step<1))
         mexErrMsgTxt("STEP values must be greater than zero.");
      }

  /* ARG 5 (optional): START */
  if (nrhs>4)
      {
      arg = prhs[4];
      if notDblMtx(arg) mexErrMsgTxt("START arg must be a double float matrix.");
      if (mxGetM(arg) * mxGetN(arg) != 2)
	mexErrMsgTxt("START arg must contain two elements.");
      mxMat = mxGetPr(arg);
      x_start = (int) mxMat[0];
      y_start = (int) mxMat[1];
      if ((x_start<1) || (x_start>x_idim) ||
          (y_start<1) || (y_start>y_idim))
         mexErrMsgTxt("START values must lie between 1 and the image dimensions.");
      }
  x_start--;  /* convert from Matlab to standard C indexes */
  y_start--;

  /* ARG 6 (optional): STOP */
  if (nrhs>5)
      {
      if notDblMtx(prhs[5]) mexErrMsgTxt("STOP arg must be double float matrix.");
      if (mxGetM(prhs[5]) * mxGetN(prhs[5]) != 2)
    	 mexErrMsgTxt("STOP arg must contain two elements.");
      mxMat = mxGetPr(prhs[5]);
      x_stop = (int) mxMat[0];
      y_stop = (int) mxMat[1];
      if ((x_stop<x_start) || (x_stop>x_idim) ||
          (y_stop<y_start) || (y_stop>y_idim))
         mexErrMsgTxt("STOP values must lie between START and the image dimensions.");
      }
  else
      {
      x_stop = x_idim;
      y_stop = y_idim;
      }
	  
  x_rdim = (x_stop-x_start+x_step-1) / x_step;
  y_rdim = (y_stop-y_start+y_step-1) / y_step;
  
  /*  mxFreeMatrix(plhs[0]); */
  plhs[0] = (mxArray *) mxCreateDoubleMatrix(x_rdim,y_rdim,mxREAL);
  if (plhs[0] == NULL) mexErrMsgTxt("Cannot allocate result matrix");
  result = mxGetPr(plhs[0]);

  temp = mxCalloc(x_fdim*y_fdim, sizeof(double));
  if (temp == NULL)
    mexErrMsgTxt("Cannot allocate necessary temporary space");

  /*
    printf("i(%d, %d), f(%d, %d), r(%d, %d), X(%d, %d, %d), Y(%d, %d, %d), %s\n",
	 x_idim,y_idim,x_fdim,y_fdim,x_rdim,y_rdim,
	 x_start,x_step,x_stop,y_start,y_step,y_stop,edges);
	 */

  /* Edited by Rajiv on April 10,2010
  if (strcmp(edges,"circular") == 0)
  	internal_wrap_reduce(image, x_idim, y_idim, filt, x_fdim, y_fdim,
			     x_start, x_step, x_stop, y_start, y_step, y_stop,
			     result);
  else internal_reduce(image, x_idim, y_idim, filt, temp, x_fdim, y_fdim,
		       x_start, x_step, x_stop, y_start, y_step, y_stop,
		       result, edges);*/

  internal_reduce(image, x_idim, y_idim, filt, temp, x_fdim, y_fdim,
		       x_start, x_step, x_stop, y_start, y_step, y_stop,
		       result, edges);
  /* End edit */

  mxFree((char *) temp);
  return;
  } 



