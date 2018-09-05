/*You can include any C libraries that you normally use*/
#include "math.h"
#include "mex.h"   /*--This one is required*/

 
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{    
    /* We have input of one double type matrix*/
    /* this function calculates the local mean, std, skewness, and kurtosis*/
    /* all at one time, for a block size of 16*/
    
    /*---Inside mexFunction---*/



    /*Declarations*/
    mxArray *xData;
    double *xVal, *outStd, *outSkw, *outKrt, *outMean;
    double mean, stdev, skw, krt, stmp, tmp, tmp1;
    int i,j,iB,jB;
    int rowLen, colLen;

    /*Copy input pointer x*/
    xData = prhs[0];

    /*Get matrix x*/
    xVal    = mxGetPr(xData);
    rowLen  = mxGetN(xData);
    colLen  = mxGetM(xData);

    /*Allocate memory and assign output pointer*/
    plhs[0] = mxCreateDoubleMatrix(colLen, rowLen, mxREAL); /*mxReal is our data-type*/
    /*Get a pointer to the data space in our newly allocated memory*/
    outStd  = mxGetPr(plhs[0]);
    
    /*Allocate memory and assign output pointer*/
    plhs[1] = mxCreateDoubleMatrix(colLen, rowLen, mxREAL); /*mxReal is our data-type*/  
    outSkw  = mxGetPr(plhs[1]);
    
    /*Allocate memory and assign output pointer*/
    plhs[2] = mxCreateDoubleMatrix(colLen, rowLen, mxREAL); /*mxReal is our data-type*/ 
    outKrt  = mxGetPr(plhs[2]);
        
	

    /*Copy matrix while multiplying each point by 2*/
    for(i=0; i<rowLen-15; i += 4)
    {
        for(j=0; j<colLen-15; j += 4)
        {
            /*Traverse through and get mean*/
            mean = 0;
            for( iB = i; iB < i+16; iB++ )
            {
                for( jB = j; jB < j+16; jB++ )
                {
                     mean += xVal[(iB*colLen)+jB];
                               
                }
            }            
            mean = mean / 256.0;
            
            /*Traverse through and get stdev, skew and kurtosis*/
            stdev = 0;
            skw   = 0;
            krt   = 0;
            for( iB = i; iB < i+16; iB++ )
            {
                for( jB = j; jB < j+16; jB++ )
                {
                     tmp =  xVal[(iB*colLen)+jB]-mean;
                     tmp1 = tmp* tmp;
                     stdev += tmp1;
                     tmp1 = tmp1 * tmp;
                     skw   += tmp1;
                     tmp1 = tmp1 * tmp;
                     krt   += tmp1;
                }
            }            
            stmp  = sqrt(stdev / 256.0);
            stdev = sqrt(stdev / 255.0);/*MATLAB's std is a bit different*/
            
            if( stmp != 0 ){
                tmp = stmp*stmp*stmp;
                tmp1 = tmp * stmp;
                skw   = (skw/256.0) /tmp ;
                krt   = (krt/256.0) /tmp1;
            }
            else{
                skw = 0;
                krt = 0;
            }
            
            for( iB = i; iB < i+4; iB++ )
            {
                for( jB = j; jB < j+4; jB++ )
                {                    
                    outStd[(iB*colLen)+jB]  = stdev;
                    outSkw[(iB*colLen)+jB]  = skw;
                    outKrt[(iB*colLen)+jB]  = krt;            
                }
            }
        }
    }


    return;
}
