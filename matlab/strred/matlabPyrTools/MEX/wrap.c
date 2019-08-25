/* 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  File: wrap.c
;;;  Author: Eero Simoncelli
;;;  Description: Circular convolution on 2D images.
;;;  Creation Date: Spring, 1987.
;;;  MODIFICATIONS:
;;;      6/96: Switched array types to double float.
;;;      2/97: made more robust and readable.  Added STOP arguments.
;;;  ----------------------------------------------------------------
;;;    Object-Based Vision and Image Understanding System (OBVIUS),
;;;      Copyright 1988, Vision Science Group,  Media Laboratory,  
;;;              Massachusetts Institute of Technology.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
*/

#include <stdlib.h>

#include "convolve.h"

/*
 --------------------------------------------------------------------
 Performs correlation (i.e., convolution with filt(-x,-y)) of FILT
 with IMAGE followed by subsampling (a.k.a. REDUCE in Burt&Adelson81).
 The operations are combined to avoid unnecessary computation of the
 convolution samples that are to be discarded in the subsampling
 operation.  The convolution is done in 9 sections so that mod
 operations are not performed unnecessarily.  The subsampling lattice
 is specified by the START, STEP and STOP parameters.
 -------------------------------------------------------------------- */

/* abstract out the inner product computation */
#define INPROD(YSTART,YIND,XSTART,XIND) \
      { \
      sum=0.0; \
      for (y_im=YSTART, filt_pos=0, x_filt_stop=x_fdim; \
	   x_filt_stop<=filt_size; \
           y_im++, x_filt_stop+=x_fdim) \
         for (x_im=XSTART ; \
    	      filt_pos<x_filt_stop; \
     	      filt_pos++, x_im++) \
   	   sum += imval[YIND][XIND] * filt[filt_pos]; \
      result[res_pos] = sum; \
      }

int internal_wrap_reduce(image, x_dim, y_dim, filt, x_fdim, y_fdim,
		     x_start, x_step, x_stop, y_start, y_step, y_stop, 
		     result)
  register image_type *filt, *result;
  register int x_dim, y_dim, x_fdim, y_fdim;
  image_type *image;
  int x_start, x_step, x_stop, y_start, y_step, y_stop;
  {
  register double sum;
  register int filt_size = x_fdim*y_fdim;
  image_type **imval;
  register int filt_pos, x_im, y_im, x_filt_stop;
  register int x_pos, y_pos, res_pos;
  int x_ctr_stop = x_dim - x_fdim + 1;
  int y_ctr_stop = y_dim - y_fdim + 1;
  int x_ctr_start = 0;
  int y_ctr_start = 0;
  int x_fmid = x_fdim/2;
  int y_fmid = y_fdim/2;
  
  /* shift start/stop coords to filter upper left hand corner */
  x_start -= x_fmid;   y_start -=  y_fmid;
  x_stop -=  x_fmid;   y_stop -=  y_fmid;

  if (x_stop < x_ctr_stop) x_ctr_stop = x_stop;
  if (y_stop < y_ctr_stop) y_ctr_stop = y_stop;

  /* Set up pointer array for rows */
  imval = (image_type **) malloc(y_dim*sizeof(image_type *));
  if (imval IS NULL)
      {
      printf("INTERNAL_WRAP: Failed to allocate temp array!");
      return(-1);
      }
  for (y_pos=y_im=0;y_pos<y_dim;y_pos++,y_im+=x_dim)
    imval[y_pos] = (image+y_im);
  
  for (res_pos=0, y_pos=y_start;	      /* TOP ROWS */
       y_pos<y_ctr_start;
       y_pos+=y_step)
    {
    for (x_pos=x_start;
	 x_pos<x_ctr_start;
	 x_pos+=x_step, res_pos++)
      INPROD(y_pos+y_dim, y_im%y_dim, x_pos+x_dim, x_im%x_dim)

    for (; 
	 x_pos<x_ctr_stop;
	 x_pos+=x_step, res_pos++) 
      INPROD(y_pos+y_dim, y_im%y_dim, x_pos, x_im)

    for (; 
	 x_pos<x_stop;
	 x_pos+=x_step, res_pos++) 
      INPROD(y_pos+y_dim, y_im%y_dim, x_pos, x_im%x_dim)
    } /* end TOP ROWS */
  
  for (;			/* MID ROWS */
       y_pos<y_ctr_stop;
       y_pos+=y_step)
    {
    for (x_pos=x_start;
	 x_pos<x_ctr_start;
	 x_pos+=x_step, res_pos++)
      INPROD(y_pos, y_im, x_pos+x_dim, x_im%x_dim)

    for (;			/* CENTER SECTION */
	 x_pos<x_ctr_stop;
	 x_pos+=x_step, res_pos++) 
      INPROD(y_pos, y_im, x_pos, x_im)

    for (; 
	 x_pos<x_stop;
	 x_pos+=x_step, res_pos++) 
      INPROD(y_pos, y_im, x_pos, x_im%x_dim)
    } /* end MID ROWS */
  
  for (;			/* BOTTOM ROWS */
       y_pos<y_stop;
       y_pos+=y_step) 
     {
    for (x_pos=x_start;
	 x_pos<x_ctr_start;
	 x_pos+=x_step, res_pos++)
      INPROD(y_pos, y_im%y_dim, x_pos+x_dim, x_im%x_dim)

    for (; 
	 x_pos<x_ctr_stop; 
	 x_pos+=x_step, res_pos++) 
      INPROD(y_pos, y_im%y_dim, x_pos, x_im)

    for (;
	 x_pos<x_stop;
	 x_pos+=x_step, res_pos++) 
      INPROD(y_pos, y_im%y_dim, x_pos, x_im%x_dim)
    } /* end BOTTOM ROWS */

  free ((image_type **) imval);

  return(0);
  }	/* end of internal_wrap_reduce */



/*
 --------------------------------------------------------------------
 Performs upsampling (padding with zeros) followed by convolution of
 FILT with IMAGE (a.k.a. EXPAND in Burt&Adelson81).  The operations
 are combined to avoid unnecessary multiplication of filter samples
 with zeros in the upsampled image.  The convolution is done in 9
 sections so that mod operation is not performed unnecessarily.
 Arguments are described in the comment above internal_wrap_reduce.

 WARNING: this subroutine destructively modifes the RESULT image, so
 the user must zero the result before invocation!
 -------------------------------------------------------------------- */

/* abstract out the inner product computation */
#define INPROD2(YSTART,YIND,XSTART,XIND) \
      { \
      val = image[im_pos]; \
      for (y_res=YSTART, filt_pos=0, x_filt_stop=x_fdim; \
	   x_filt_stop<=filt_size; \
	   y_res++, x_filt_stop+=x_fdim) \
	for (x_res=XSTART; \
	     filt_pos<x_filt_stop; \
	     filt_pos++, x_res++) \
	  imval[YIND][XIND] += val * filt[filt_pos]; \
      }

int internal_wrap_expand(image, filt, x_fdim, y_fdim,
	      x_start, x_step, x_stop, y_start, y_step, y_stop,
	      result, x_dim, y_dim)
  register image_type *filt, *result;
  register int x_fdim, y_fdim, x_dim, y_dim;
  image_type *image; 
  int x_start, x_step, x_stop, y_start, y_step, y_stop;
  {
  register double val;
  register int filt_size = x_fdim*y_fdim;
  image_type **imval;
  register int filt_pos, x_res, y_res, x_filt_stop;
  register int x_pos, y_pos, im_pos;
  int x_ctr_stop = x_dim - x_fdim + 1;
  int y_ctr_stop = y_dim - y_fdim + 1;
  int x_ctr_start = 0;
  int y_ctr_start = 0;
  int x_fmid = x_fdim/2;
  int y_fmid = y_fdim/2;
  
  /* shift start/stop coords to filter upper left hand corner */
  x_start -= x_fmid;   y_start -=  y_fmid;
  x_stop -=  x_fmid;   y_stop -=  y_fmid;

  if (x_stop < x_ctr_stop) x_ctr_stop = x_stop;
  if (y_stop < y_ctr_stop) y_ctr_stop = y_stop;

  /* Set up pointer array for rows */
  imval = (image_type **) malloc(y_dim*sizeof(image_type *));
  if (imval IS NULL)
      {
      printf("INTERNAL_WRAP: Failed to allocate temp array!");
      return(-1);
      }
  for (y_pos=y_res=0;y_pos<y_dim;y_pos++,y_res+=x_dim)
    imval[y_pos] = (result+y_res);
  
  for (im_pos=0, y_pos=y_start;	/* TOP ROWS */
       y_pos<y_ctr_start;
       y_pos+=y_step)
    {
    for (x_pos=x_start; 
	 x_pos<x_ctr_start;
	 x_pos+=x_step, im_pos++)
      INPROD2(y_pos+y_dim, y_res%y_dim, x_pos+x_dim, x_res%x_dim)
	
    for (; 
	 x_pos<x_ctr_stop;
	 x_pos+=x_step, im_pos++) 
      INPROD2(y_pos+y_dim, y_res%y_dim, x_pos, x_res)

    for (; 
	 x_pos<x_stop;
	 x_pos+=x_step, im_pos++) 
      INPROD2(y_pos+y_dim, y_res%y_dim, x_pos, x_res%x_dim)
    } /* end TOP ROWS */
  
  for (;			/* MID ROWS */
       y_pos<y_ctr_stop;
       y_pos+=y_step)
    {
    for (x_pos=x_start; 
	 x_pos<x_ctr_start;
	 x_pos+=x_step, im_pos++)
      INPROD2(y_pos, y_res, x_pos+x_dim, x_res%x_dim)
	
    for (;			/* CENTER SECTION */
	 x_pos<x_ctr_stop;
	 x_pos+=x_step, im_pos++) 
      INPROD2(y_pos, y_res, x_pos, x_res)

    for (; 
	 x_pos<x_stop;
	 x_pos+=x_step, im_pos++) 
      INPROD2(y_pos, y_res, x_pos, x_res%x_dim)
    } /* end MID ROWS */
  
  for (;			/* BOTTOM ROWS */
       y_pos<y_stop;
       y_pos+=y_step) 
    {
    for (x_pos=x_start; 
	 x_pos<x_ctr_start;
	 x_pos+=x_step, im_pos++)
      INPROD2(y_pos, y_res%y_dim, x_pos+x_dim, x_res%x_dim)
	
    for (; 
	 x_pos<x_ctr_stop;
	 x_pos+=x_step, im_pos++) 
      INPROD2(y_pos, y_res%y_dim, x_pos, x_res)

    for (; 
	 x_pos<x_stop;
	 x_pos+=x_step, im_pos++) 
      INPROD2(y_pos, y_res%y_dim, x_pos, x_res%x_dim)
    } /* end BOTTOM ROWS */

  free ((image_type **) imval);
  return(0);
  } /* end of internal_wrap_expand */



/* Local Variables: */
/* buffer-read-only: t */
/* End: */
