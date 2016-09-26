#ifndef M2PYTHONTOOLS_H_

#include <stdio.h>
#include <ctype.h>

// Distinguish between M2k and OpenContainers usage
#if defined(OC_USE_OC) || defined(OC_NEW_STYLE_INCLUDES)
# include <ocarray.h>

PTOOLS_BEGIN_NAMESPACE

#else

# include "m2opalprotocol2.h"
# include "m2array.h"

inline bool IsLittleEndian ()
{ // Set the endian-ness of the machine we are depickling to ...      
  int_4 testme = 0x00000001;
  char* testme_ptr = (char*)&testme;
  bool little_endian = (*testme_ptr == 0x01);
  return little_endian;
}

#endif

// ///// Helper Functions


inline void InPlaceReEndianize (char* buffer, int buffer_elements, 
				int element_length, bool is_cx)
{
  if (is_cx) { 
    buffer_elements *= 2;
    element_length  /= 2;
  }
  for (int ii=0; ii<buffer_elements; ii++) {
    for (int jj=0; jj<element_length/2; jj++) {
      const int oo = element_length-jj-1;
      char temp = buffer[jj];
      buffer[jj]     = buffer[oo];
      buffer[oo]     = temp;
    }
    buffer+=element_length;
  }
}

// Helper function for Python Depickling:
// "Print" from string to contiguous memory of a Vector.
int CopyPrintableBufferToVector (const char* print_buff, size_t pb_bytes,
				 char* v, size_t v_bytes);
// Helper function for Python Pickler
void PrintBufferToString (const char* vbuff, int bytes, Array<char>& a);




// Distinguish between M2k and OpenContainers usage
#if defined(OC_USE_OC) || defined(OC_NEW_STYLE_INCLUDES)
PTOOLS_END_NAMESPACE
#endif

#define M2PYTHONTOOLS_H_
#endif // M2PYTHONTOOLS_H_
