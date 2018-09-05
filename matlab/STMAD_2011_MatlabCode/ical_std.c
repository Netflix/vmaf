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
    mxArray *xData,*yData;
    double *xVal, *yVal, *outStd, *outStdMod, *outMean;
    double mean, mean2, stdev, tmp1;
    double *TMP;
    int i,j,iB,jB;
    int rowLen, colLen;

    /*Copy input pointer x*/
    xData = prhs[0];
    yData = prhs[1];

    /*Get matrix x*/
    xVal = mxGetPr(xData);
    rowLen  = mxGetN(xData);
    colLen  = mxGetM(xData);
    
    /*Get matrix y*/
    yVal = mxGetPr(yData);
    rowLen  = mxGetN(yData);
    colLen  = mxGetM(yData);

    /*Allocate memory and assign output pointer*/
    plhs[0] = mxCreateDoubleMatrix(colLen, rowLen, mxREAL); /*mxReal is our data-type*/
    /*Get a pointer to the data space in our newly allocated memory*/
    outStd  = mxGetPr(plhs[0]);
    
    /*Allocate memory and assign output pointer*/
    plhs[1] = mxCreateDoubleMatrix(colLen, rowLen, mxREAL); /*mxReal is our data-type*/  
    outStdMod  = mxGetPr(plhs[1]);
    
    /*Allocate memory and assign output pointer*/
    plhs[2] = mxCreateDoubleMatrix(colLen, rowLen, mxREAL); /*mxReal is our data-type*/   
    outMean  = mxGetPr(plhs[2]);
      
    TMP = mxGetPr( mxCreateDoubleMatrix(colLen, rowLen, mxREAL) );

    /*Copy matrix while multiplying each point by 2*/
    for(i=0; i<rowLen-15; i += 4)
    {
        for(j=0; j<colLen-15; j += 4)
        {
            /*Traverse through and get mean*/
            mean = 0;
            mean2= 0;
            for( iB = i; iB < i+16; iB++ )
            {
                for( jB = j; jB < j+16; jB++ )
                {
                     mean +=  xVal[(iB*colLen)+jB];
                     mean2 += yVal[(iB*colLen)+jB];
                               
                }
            }            
            mean = mean / 256.0;
            mean2= mean2 / 256.0;
            
            /*Traverse through and get stdev*/
            stdev = 0;            
            for( iB = i; iB < i+16; iB++ )
            {
                for( jB = j; jB < j+16; jB++ )
                {
                     tmp1 = xVal[(iB*colLen)+jB]-mean;
                     stdev += tmp1 * tmp1;
                }
            }                        
            stdev = sqrt(stdev / 255.0);/*MATLAB's std is a bit different*/                     
            
            for( iB = i; iB < i+4; iB++ )
            {
                for( jB = j; jB < j+4; jB++ )
                {
                    outMean[(iB*colLen)+jB] = mean2;/* mean of reference*/
                    outStd[(iB*colLen)+jB]  = stdev;/* stdev of dst*/
                }
            }                       
        }
    }
    
    /*====================================================================*/
    /*Modified STD*/
        
    for(i=0; i<rowLen-15; i += 4)
    {
        for(j=0; j<colLen-15; j += 4)
        {
            /*Traverse through and get mean*/
            mean = 0;
            
            for( iB = i; iB < i+8; iB ++ )
            {
                for( jB = j; jB < j+8; jB ++ )
                {
                     mean +=  yVal[(iB*colLen)+jB];                                                    
                }
            }            
            mean = mean / 64.0;            
            
            /*Traverse through and get stdev*/
            stdev = 0;            
            for( iB = i; iB < i+8; iB++ )
            {
                for( jB = j; jB < j+8; jB++ )
                {
                     tmp1 = yVal[(iB*colLen)+jB]-mean;
                     stdev += tmp1 * tmp1; 
                }
            }                        
            stdev = sqrt(stdev / 63.0);/*MATLAB's std is a bit different*/                       
            
            for( iB = i; iB < i+4; iB++ )
            {
                for( jB = j; jB < j+4; jB++ )
                {                    
                    TMP[(iB*colLen)+jB]  = stdev;/* stdev of ref*/ 
                    outStdMod[(iB*colLen)+jB] = stdev;
                }
            }                       
        }
    }
    
    for(i=0; i<rowLen-15; i += 4)
    {        
        for(j=0; j<colLen-15; j += 4)
        {
            mean = TMP[(i*colLen)+j];
            for( iB = i; iB < i+8; iB += 5 )
            {
                for( jB = j; jB < j+8; jB += 5 )
                {
                   if( iB < rowLen-15 && jB < colLen-15 && mean > TMP[(iB*colLen)+jB]  )
                       mean = TMP[(iB*colLen)+jB];
                }
            }
                       
            for( iB = i; iB < i+4; iB++ )
            {
                for( jB = j; jB < j+4; jB++ )
                {
                     outStdMod[(iB*colLen)+jB] = mean;                                                   
                }
            }     
        }
    }
    mxDestroyArray(TMP);

    return;
}
        
