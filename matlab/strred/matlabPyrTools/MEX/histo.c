/* 
[N, X] = histo(MTX, NBINS_OR_BINSIZE, BIN_CENTER)
  >>> See histo.m for documentation <<<
  EPS, ported from OBVIUS, 3/97.
*/

#define V4_COMPAT
#include <matrix.h>  /* Matlab matrices */
#include <mex.h>

#include <stddef.h>  /* NULL */
#include <math.h>  /* ceil */

#define notDblMtx(it) (!mxIsNumeric(it) || !mxIsDouble(it) || mxIsSparse(it) || mxIsComplex(it))

#define PAD 0.49999 /* A hair below 1/2, to avoid roundoff errors */
#define MAXBINS 20000

void mexFunction(int nlhs,	     /* Num return vals on lhs */
		 mxArray *plhs[],    /* Matrices on lhs      */
		 int nrhs,	     /* Num args on rhs    */
		 const mxArray *prhs[]     /* Matrices on rhs */
		 )
  {
  register double temp;
  register int binnum, i, size;
  register double *im, binsize;
  register double origin, *hist, mn, mx, mean;
  register int nbins;
  double *bincenters; 
  mxArray *arg;
  double *mxMat;

  if (nrhs < 1 ) mexErrMsgTxt("requires at least 1 argument.");

  /* ARG 1: MATRIX  */
  arg = prhs[0];
  if notDblMtx(arg) mexErrMsgTxt("MTX arg must be a real non-sparse matrix.");
  im = mxGetPr(arg);
  size = (int) mxGetM(arg) * mxGetN(arg);

  /* FIND min, max, mean values of MTX */
  mn = *im;   mx = *im;  binsize = 0;
  for (i=1; i<size; i++)
    {
      temp = im[i];
      if (temp < mn)
	mn = temp;
      else if (temp > mx)
	mx = temp;
      binsize += temp;
    }
  mean = binsize / size;

  /* ARG 3: BIN_CENTER */
  if (nrhs > 2)
    {
    arg = prhs[2];
    if notDblMtx(arg) mexErrMsgTxt("BIN_CENTER arg must be a real scalar.");
    if (mxGetM(arg) * mxGetN(arg) != 1)
      mexErrMsgTxt("BIN_CENTER must be a real scalar.");
    mxMat= mxGetPr(arg);
    origin = *mxMat;
    }
  else
    origin = mean;

  /* ARG 2: If positive, NBINS.  If negative, -BINSIZE. */
  if (nrhs > 1)
    {
    arg = prhs[1];
    if notDblMtx(arg) mexErrMsgTxt("NBINS_OR_BINSIZE arg must be a real scalar.");
    if (mxGetM(arg) * mxGetN(arg) != 1)
      mexErrMsgTxt("NBINS_OR_BINSIZE must be a real scalar.");
    mxMat= mxGetPr(arg);
    binsize = *mxMat;
    }
  else
    {
    binsize = 101;  /* DEFAULT: 101 bins */
    }

  /* --------------------------------------------------
     Adjust origin, binsize, nbins such that
        mx <= origin + (nbins-1)*binsize + PAD*binsize
	mn >= origin - PAD*binsize
     -------------------------------------------------- */
  if (binsize < 0)		/* user specified BINSIZE */
      {
      binsize = -binsize;
      origin -= binsize * ceil((origin-mn-PAD*binsize)/binsize);
      nbins = (int) ceil((mx-origin-PAD*binsize)/binsize) + 1;
      }
  else				/* user specified NBINS */
      {
      nbins = (int) (binsize + 0.5);    /* round to int */
      if (nbins == 0)
	mexErrMsgTxt("NBINS must be greater than zero.");
      binsize = (mx-mn)/(nbins-1+2*PAD);   /* start with lower bound */
      i = ceil((origin-mn-binsize/2)/binsize);
      if ( mn < (origin-i*binsize-PAD*binsize) )
	binsize = (origin-mn)/(i+PAD);
      else if ( mx > (origin+(nbins-1-i)*binsize+PAD*binsize) )
	binsize = (mx-origin)/((nbins-1-i)+PAD);
      origin -= binsize * ceil((origin-mn-PAD*binsize)/binsize);
      }

  if (nbins > MAXBINS)
      {
      mexPrintf("nbins: %d,  MAXBINS: %d\n",nbins,MAXBINS);
      mexErrMsgTxt("Number of histo bins has exceeded maximum");
      }

  /* Allocate hist  and xvals */
  plhs[0] = (mxArray *) mxCreateDoubleMatrix(1,nbins,mxREAL);
  if (plhs[0] == NULL) mexErrMsgTxt("Error allocating result matrix");
  hist = mxGetPr(plhs[0]);

  if (nlhs > 1)
      {
      plhs[1] = (mxArray *) mxCreateDoubleMatrix(1,nbins,mxREAL);
      if (plhs[1] == NULL) mexErrMsgTxt("Error allocating result matrix");
      bincenters = mxGetPr(plhs[1]);
      for (i=0, temp=origin; i<nbins; i++, temp+=binsize)
	bincenters[i] = temp;
      }

  for (i=0; i<size; i++)
      {
      binnum = (int) ((im[i] - origin)/binsize + 0.5);
      if ((binnum < nbins) && (binnum >= 0))
	(hist[binnum]) += 1.0;
      else
	printf("HISTO warning: value %f outside of range [%f,%f]\n",
	       im[i], origin-0.5*binsize, origin+(nbins-0.5)*binsize);
      }

  return;
  }      

