/* 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  File: convolve.h
;;;  Author: Simoncelli
;;;  Description: Header file for convolve.c
;;;  Creation Date:
;;;  ----------------------------------------------------------------
;;;    Object-Based Vision and Image Understanding System (OBVIUS),
;;;      Copyright 1988, Vision Science Group,  Media Laboratory,  
;;;              Massachusetts Institute of Technology.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
*/

#include <stdio.h>
#include <stdlib.h> 

#define ABS(x)	  (((x)>=0) ? (x) : (-(x)))
#define ROOT2 1.4142135623730951
#define REDUCE 0
#define EXPAND 1
#define IS    ==
#define ISNT  !=
#define AND &&
#define OR ||

typedef  int (*fptr)();

typedef struct 
  {
  char *name;
  fptr func;
  } EDGE_HANDLER;

typedef double image_type;

fptr edge_function(char *edges);
int internal_reduce(image_type *image, int x_idim, int y_idim, 
		    image_type *filt, image_type *temp, int x_fdim, int y_fdim,
		    int x_start, int x_step, int x_stop, 
		    int y_start, int y_step, int y_stop,
		    image_type *result, char *edges);
int internal_expand(image_type *image, 
		    image_type *filt, image_type *temp, int x_fdim, int y_fdim,
		    int x_start, int x_step, int x_stop, 
		    int y_start, int y_step, int y_stop,
		    image_type *result, int x_rdim, int y_rdim, char *edges);
int internal_wrap_reduce(image_type *image, int x_idim, int y_idim, 
			 image_type *filt, int x_fdim, int y_fdim,
			 int x_start, int x_step, int x_stop, 
			 int y_start, int y_step, int y_stop,
			 image_type *result);
int internal_wrap_expand(image_type *image, image_type *filt, int x_fdim, int y_fdim,
			 int x_start, int x_step, int x_stop, 
			 int y_start, int y_step, int y_stop,
			 image_type *result, int x_rdim, int y_rdim);
