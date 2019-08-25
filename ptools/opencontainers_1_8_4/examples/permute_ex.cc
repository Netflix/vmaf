
// Example how to use the permute.  The ocpermute.h is straight, inline C code,
// so that file has no include dependencies.

#include "ocpermute.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// Helper routine for printing permutations
void printarray(int *a, int len)
{
  for (int ii=0; ii<len; ii++) {
    if (ii<len-1) {
      printf("%d ", a[ii]);
    } else {
      printf("%d", a[ii]);
    }
    
  }
  printf("\n");
}

void usage (char** argv)
{
  fprintf(stderr, "%s [n] [--timing]\n", argv[0]);
  fprintf(stderr, "  With --timing, no arrays are printed so you can time\n");
  exit(1);
}

int main (int argc, char **argv)
{
  int no_output = 0;

  if (argc>4) {
    usage(argv);
  }

  // Do we do output loop or timing loop?
  if (argc==3) {
    if (strcmp(argv[2], "--timing") == 0) {
      no_output = 1;
    } else {
      usage(argv);
    }
  }

  // Allocate and initialize array
  int len = (argc>1) ? atoi(argv[1]) : 4;
  int* a = (int*)malloc(len*sizeof(int));
  for (int ii=0; ii<len; ii++) {
    a[ii] = ii;
  }

  // Print each permutation, count em!
  // Why two loops?  So you don't pay for the "if no_output"
  // at each iteration.
  int count = 0;
  if (no_output) {
    do {
      count += 1;
      //printarray(a, len);
    } while (perm(a, len));

  } else {
    do {
      count += 1;
      printarray(a, len);
    } while (perm(a, len));
  }

  printf("Total Permutations:%d\n", count);

}

