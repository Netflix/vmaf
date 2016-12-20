/* 
RES = innerProd(MAT);
  Computes mat'*mat  
  Odelia Schwartz, 8/97.
*/

#define V4_COMPAT
#include <matrix.h>  

#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <strings.h>
#include <stdlib.h>

void mexFunction(int nlhs,           /* Num return vals on lhs */
                 mxArray *plhs[],    /* Matrices on lhs      */
                 int nrhs,           /* Num args on rhs    */
                 const mxArray *prhs[]     /* Matrices on rhs */
                 )
{
   register double *res, *mat, tmp;
   register int len, wid, i, k, j, jlen, ilen, imat, jmat;
   mxArray *arg;
   
   /* get matrix input argument */
   /* should be matrix in which num rows >= num columns */
   arg=prhs[0];                     
   mat= mxGetPr(arg);
   len = (int) mxGetM(arg);
   wid = (int) mxGetN(arg);
   if ( wid > len )
     printf("innerProd: Warning: width %d is greater than length %d.\n",wid,len); 
   plhs[0] = (mxArray *) mxCreateDoubleMatrix(wid,wid,mxREAL);
   if (plhs[0] == NULL) 
     mexErrMsgTxt(sprintf("Error allocating %dx%d result matrix",wid,wid));
   res = mxGetPr(plhs[0]);

   for(i=0, ilen=0; i<wid; i++, ilen+=len)
     {
      for(j=i, jlen=ilen; j<wid; j++, jlen+=len)
         {
   	    tmp = 0.0;
            for(k=0, imat=ilen, jmat=jlen; k<len; k++, imat++, jmat++)
	      tmp += mat[imat]*mat[jmat];
	    res[i*wid+j] = tmp;
            res[j*wid+i] = tmp;
         }
   }
   return;

}
