/* 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  File: convolve.c
;;;  Author: Eero Simoncelli
;;;  Description: General convolution code for 2D images
;;;  Creation Date: Spring, 1987.
;;;  MODIFICATIONS:
;;;     10/89: approximately optimized the choice of register vars on SPARCS.
;;;      6/96: Switched array types to double float.
;;;      2/97: made more robust and readable.  Added STOP arguments.
;;;      8/97: Bug: when calling internal_reduce with edges in {reflect1,repeat,
;;;            extend} and an even filter dimension.  Solution: embed the filter
;;;            in the upper-left corner of a filter with odd Y and X dimensions.
;;;  ----------------------------------------------------------------
;;;    Object-Based Vision and Image Understanding System (OBVIUS),
;;;      Copyright 1988, Vision Science Group,  Media Laboratory,  
;;;              Massachusetts Institute of Technology.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
*/

#include <stdio.h>
#include <math.h>
#include "convolve.h"

/*
  --------------------------------------------------------------------
  Correlate FILT with IMAGE, subsampling according to START, STEP, and
  STOP parameters, with values placed into RESULT array.  RESULT
  dimensions should be ceil((stop-start)/step).  TEMP should be a
  pointer to a temporary double array the size of the filter.
  EDGES is a string specifying how to handle boundaries -- see edges.c.
  The convolution is done in 9 sections, where the border sections use
  specially computed edge-handling filters (see edges.c). The origin 
  of the filter is assumed to be (floor(x_fdim/2), floor(y_fdim/2)).
------------------------------------------------------------------------ */

/* abstract out the inner product computation */
#define INPROD(XCNR,YCNR)  \
        { \
	sum=0.0; \
	for (im_pos=YCNR*x_dim+XCNR, filt_pos=0, x_filt_stop=x_fdim; \
	     x_filt_stop<=filt_size; \
	     im_pos+=(x_dim-x_fdim), x_filt_stop+=x_fdim) \
	    for (; \
		 filt_pos<x_filt_stop; \
		 filt_pos++, im_pos++) \
	      sum+= image[im_pos]*temp[filt_pos]; \
        result[res_pos] = sum; \
	}

int internal_reduce(image, x_dim, y_dim, filt, temp, x_fdim, y_fdim,
		x_start, x_step, x_stop, y_start, y_step, y_stop,
		result, edges)
  register image_type *image, *temp;
  register int x_fdim, x_dim;
  register image_type *result;
  register int x_step, y_step;
  int x_start, y_start;
  int x_stop, y_stop;     
  image_type *filt; 
  int y_dim, y_fdim;
  char *edges;
  { 
  register double sum;
  register int filt_pos, im_pos, x_filt_stop;
  register int x_pos, filt_size = x_fdim*y_fdim;
  register int y_pos, res_pos;
  register int y_ctr_stop = y_dim - ((y_fdim==1)?0:y_fdim);
  register int x_ctr_stop = x_dim - ((x_fdim==1)?0:x_fdim);
  register int x_res_dim = (x_stop-x_start+x_step-1)/x_step;
  int x_ctr_start = ((x_fdim==1)?0:1);
  int y_ctr_start = ((y_fdim==1)?0:1);
  int x_fmid = x_fdim/2;
  int y_fmid = y_fdim/2;
  int base_res_pos;
  fptr reflect = edge_function(edges);  /* look up edge-handling function */

  if (!reflect) return(-1);

  /* shift start/stop coords to filter upper left hand corner */
  x_start -= x_fmid;   y_start -=  y_fmid;
  x_stop -=  x_fmid;   y_stop -=  y_fmid;

  if (x_stop < x_ctr_stop) x_ctr_stop = x_stop;
  if (y_stop < y_ctr_stop) y_ctr_stop = y_stop;

  for (res_pos=0, y_pos=y_start;	      /* TOP ROWS */
       y_pos<y_ctr_start;
       y_pos+=y_step)
    {
    for (x_pos=x_start;			      /* TOP-LEFT CORNER */
	 x_pos<x_ctr_start;
	 x_pos+=x_step, res_pos++)
      {
      (*reflect)(filt,x_fdim,y_fdim,x_pos-1,y_pos-1,temp,REDUCE);
      INPROD(0,0)
      }

    (*reflect)(filt,x_fdim,y_fdim,0,y_pos-1,temp,REDUCE);
    for (;				      /* TOP EDGE */
	 x_pos<x_ctr_stop;
	 x_pos+=x_step, res_pos++) 
      INPROD(x_pos,0)

    for (;				      /* TOP-RIGHT CORNER */
	 x_pos<x_stop;
	 x_pos+=x_step, res_pos++) 
      {
      (*reflect)(filt,x_fdim,y_fdim,x_pos-x_ctr_stop+1,y_pos-1,temp,REDUCE);
      INPROD(x_ctr_stop,0)
      }
    } /* end TOP ROWS */   

  y_ctr_start = y_pos;			      /* hold location of top */
  for (base_res_pos=res_pos, x_pos=x_start;   /* LEFT EDGE */
       x_pos<x_ctr_start;
       x_pos+=x_step, base_res_pos++)
    {
    (*reflect)(filt,x_fdim,y_fdim,x_pos-1,0,temp,REDUCE);
    for (y_pos=y_ctr_start, res_pos=base_res_pos;
	 y_pos<y_ctr_stop;
	 y_pos+=y_step, res_pos+=x_res_dim)
      INPROD(0,y_pos)
    }

  (*reflect)(filt,x_fdim,y_fdim,0,0,temp,REDUCE);
  for (;				      /* CENTER */
       x_pos<x_ctr_stop;
       x_pos+=x_step, base_res_pos++) 
    for (y_pos=y_ctr_start, res_pos=base_res_pos;
	 y_pos<y_ctr_stop;
	 y_pos+=y_step, res_pos+=x_res_dim)
      INPROD(x_pos,y_pos)

  for (;				      /* RIGHT EDGE */
       x_pos<x_stop;
       x_pos+=x_step, base_res_pos++)
    {
    (*reflect)(filt,x_fdim,y_fdim,x_pos-x_ctr_stop+1,0,temp,REDUCE);
    for (y_pos=y_ctr_start, res_pos=base_res_pos;
	 y_pos<y_ctr_stop;
	 y_pos+=y_step, res_pos+=x_res_dim)
      INPROD(x_ctr_stop,y_pos)
    }

  for (res_pos-=(x_res_dim-1);
       y_pos<y_stop;			      /* BOTTOM ROWS */
       y_pos+=y_step) 
    {
    for (x_pos=x_start;			      /* BOTTOM-LEFT CORNER */
	 x_pos<x_ctr_start;
	 x_pos+=x_step, res_pos++)
      {
      (*reflect)(filt,x_fdim,y_fdim,x_pos-1,y_pos-y_ctr_stop+1,temp,REDUCE);
      INPROD(0,y_ctr_stop)
      }

    (*reflect)(filt,x_fdim,y_fdim,0,y_pos-y_ctr_stop+1,temp,REDUCE);
    for (;				      /* BOTTOM EDGE */
	 x_pos<x_ctr_stop;
	 x_pos+=x_step, res_pos++) 
      INPROD(x_pos,y_ctr_stop)

    for (;				      /* BOTTOM-RIGHT CORNER */
	 x_pos<x_stop;
	 x_pos+=x_step, res_pos++) 
      {
      (*reflect)(filt,x_fdim,y_fdim,x_pos-x_ctr_stop+1,y_pos-y_ctr_stop+1,temp,REDUCE);
      INPROD(x_ctr_stop,y_ctr_stop)
      }
    } /* end BOTTOM */
  return(0);
  } /* end of internal_reduce */


/*
  --------------------------------------------------------------------
  Upsample IMAGE according to START,STEP, and STOP parameters and then
  convolve with FILT, adding values into RESULT array.  IMAGE
  dimensions should be ceil((stop-start)/step).  See
  description of internal_reduce (above).

  WARNING: this subroutine destructively modifies the RESULT array!
 ------------------------------------------------------------------------ */

/* abstract out the inner product computation */
#define INPROD2(XCNR,YCNR)  \
        { \
        val = image[im_pos]; \
	for (res_pos=YCNR*x_dim+XCNR, filt_pos=0, x_filt_stop=x_fdim; \
	     x_filt_stop<=filt_size; \
	     res_pos+=(x_dim-x_fdim), x_filt_stop+=x_fdim) \
	    for (; \
		 filt_pos<x_filt_stop; \
		 filt_pos++, res_pos++) \
	      result[res_pos] += val*temp[filt_pos]; \
	}

int internal_expand(image,filt,temp,x_fdim,y_fdim,
		x_start,x_step,x_stop,y_start,y_step,y_stop,
		result,x_dim,y_dim,edges)
  register image_type *result, *temp;
  register int x_fdim, x_dim;
  register int x_step, y_step;
  register image_type *image; 
  int x_start, y_start;
  image_type *filt; 
  int y_fdim, y_dim;
  char *edges;
  {
  register double val;
  register int filt_pos, res_pos, x_filt_stop;
  register int x_pos, filt_size = x_fdim*y_fdim;
  register int y_pos, im_pos;
  register int x_ctr_stop = x_dim - ((x_fdim==1)?0:x_fdim);
  int y_ctr_stop = (y_dim - ((y_fdim==1)?0:y_fdim));
  int x_ctr_start = ((x_fdim==1)?0:1);
  int y_ctr_start = ((y_fdim==1)?0:1);
  int x_fmid = x_fdim/2;
  int y_fmid = y_fdim/2;
  int base_im_pos, x_im_dim = (x_stop-x_start+x_step-1)/x_step;
  fptr reflect = edge_function(edges);  /* look up edge-handling function */	 

  if (!reflect) return(-1);

  /* shift start/stop coords to filter upper left hand corner */
  x_start -= x_fmid;   y_start -=  y_fmid;
  x_stop -=  x_fmid;   y_stop -=  y_fmid;

  if (x_stop < x_ctr_stop) x_ctr_stop = x_stop;
  if (y_stop < y_ctr_stop) y_ctr_stop = y_stop;

  for (im_pos=0, y_pos=y_start;		      /* TOP ROWS */
       y_pos<y_ctr_start;
       y_pos+=y_step)
    {
    for (x_pos=x_start;			      /* TOP-LEFT CORNER */
	 x_pos<x_ctr_start;
	 x_pos+=x_step, im_pos++)
      {
      (*reflect)(filt,x_fdim,y_fdim,x_pos-1,y_pos-1,temp,EXPAND);
      INPROD2(0,0)
      }

    (*reflect)(filt,x_fdim,y_fdim,0,y_pos-1,temp,EXPAND);
    for (;				      /* TOP EDGE */
	 x_pos<x_ctr_stop;
	 x_pos+=x_step, im_pos++) 
      INPROD2(x_pos,0)

    for (;				      /* TOP-RIGHT CORNER */
	 x_pos<x_stop;
	 x_pos+=x_step, im_pos++) 
      {
      (*reflect)(filt,x_fdim,y_fdim,x_pos-x_ctr_stop+1,y_pos-1,temp,EXPAND);
      INPROD2(x_ctr_stop,0)
      }
    }                                           /* end TOP ROWS */   

  y_ctr_start = y_pos;			      /* hold location of top */
  for (base_im_pos=im_pos, x_pos=x_start;     /* LEFT EDGE */
       x_pos<x_ctr_start;
       x_pos+=x_step, base_im_pos++)
    {
    (*reflect)(filt,x_fdim,y_fdim,x_pos-1,0,temp,EXPAND);
    for (y_pos=y_ctr_start, im_pos=base_im_pos;
	 y_pos<y_ctr_stop;
	 y_pos+=y_step, im_pos+=x_im_dim)
      INPROD2(0,y_pos)
    }

  (*reflect)(filt,x_fdim,y_fdim,0,0,temp,EXPAND);
  for (;				      /* CENTER */
       x_pos<x_ctr_stop;
       x_pos+=x_step, base_im_pos++) 
    for (y_pos=y_ctr_start, im_pos=base_im_pos;
	 y_pos<y_ctr_stop;
	 y_pos+=y_step, im_pos+=x_im_dim)
      INPROD2(x_pos,y_pos)

  for (;				      /* RIGHT EDGE */
       x_pos<x_stop;
       x_pos+=x_step, base_im_pos++)
    {
    (*reflect)(filt,x_fdim,y_fdim,x_pos-x_ctr_stop+1,0,temp,EXPAND);
    for (y_pos=y_ctr_start, im_pos=base_im_pos;
	 y_pos<y_ctr_stop;
	 y_pos+=y_step, im_pos+=x_im_dim)
      INPROD2(x_ctr_stop,y_pos)
    }  

  for (im_pos-=(x_im_dim-1);
       y_pos<y_stop;			      /* BOTTOM ROWS */
       y_pos+=y_step) 
    {
    for (x_pos=x_start;			      /* BOTTOM-LEFT CORNER */
	 x_pos<x_ctr_start;
	 x_pos+=x_step, im_pos++)
      {
      (*reflect)(filt,x_fdim,y_fdim,x_pos-1,y_pos-y_ctr_stop+1,temp,EXPAND);
      INPROD2(0,y_ctr_stop)
      }

    (*reflect)(filt,x_fdim,y_fdim,0,y_pos-y_ctr_stop+1,temp,EXPAND);
    for (;				      /* BOTTOM EDGE */
	 x_pos<x_ctr_stop;
	 x_pos+=x_step, im_pos++) 
      INPROD2(x_pos,y_ctr_stop)

    for (;				      /* BOTTOM-RIGHT CORNER */
	 x_pos<x_stop;
	 x_pos+=x_step, im_pos++) 
      {
      (*reflect)(filt,x_fdim,y_fdim,x_pos-x_ctr_stop+1,y_pos-y_ctr_stop+1,temp,EXPAND);
      INPROD2(x_ctr_stop,y_ctr_stop)
      }
    } /* end BOTTOM */
  return(0);
  } /* end of internal_expand */


/* Local Variables: */
/* buffer-read-only: t */
/* End: */

