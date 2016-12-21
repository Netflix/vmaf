/* 
[MIN, MAX] = range2(MTX)
  >>> See range2.m for documentation <<<
  EPS, 3/97.
*/

#define V4_COMPAT
#include <matrix.h>  /* Matlab matrices */
#include <mex.h>

#include <stddef.h>  /* NULL */

#define notDblMtx(it) (!mxIsNumeric(it) || !mxIsDouble(it) || mxIsSparse(it) || mxIsComplex(it))

void mexFunction(int nlhs,	     /* Num return vals on lhs */
		 mxArray *plhs[],    /* Matrices on lhs      */
		 int nrhs,	     /* Num args on rhs    */
		 const mxArray *prhs[]     /* Matrices on rhs */
		 )
  {
  register double temp, mn, mx;
  register double *mtx;
  register int i, size;
  mxArray *arg;

  if (nrhs != 1) mexErrMsgTxt("requires 1 argument.");

  /* ARG 1: MATRIX  */
  arg = prhs[0];
  if notDblMtx(arg) mexErrMsgTxt("MTX arg must be a real non-sparse matrix.");
  mtx = mxGetPr(arg);
  size = (int) mxGetM(arg) * mxGetN(arg);

  /* FIND min, max values of MTX */
  mn = *mtx;   mx = *mtx;  
  for (i=1; i<size; i++)
      {
      temp = mtx[i];
      if (temp < mn)
	mn = temp;
      else if (temp > mx)
	mx = temp;
      }

  plhs[0] = (mxArray *) mxCreateDoubleMatrix(1,1,mxREAL);
  if (plhs[0] == NULL) mexErrMsgTxt("Error allocating result matrix");
  plhs[1] = (mxArray *) mxCreateDoubleMatrix(1,1,mxREAL);
  if (plhs[1] == NULL) mexErrMsgTxt("Error allocating result matrix");
  mtx = mxGetPr(plhs[0]);
  mtx[0] = mn;
  mtx = mxGetPr(plhs[1]);
  mtx[0] = mx;

  return;
  }      

