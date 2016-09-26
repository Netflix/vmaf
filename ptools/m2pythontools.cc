
#include "m2pythontools.h"
#include <string.h>  // for strchr

// Distinguish between M2k and OpenContainers usage
#if defined(OC_USE_OC) || defined(OC_NEW_STYLE_INCLUDES)
PTOOLS_BEGIN_NAMESPACE
#endif

// Unfortunately, there is some code replication from
// opencontainers_1_4_4 and here for the printing routines below: this
// is because the Midas 2k routines need this as well, and they don't
// use OpenContainers

// Helper function for Depickling: Take an ascii string with \x## and
// other escape strings, and turn it into the binary string/vector
// those escapes refer to.  This returns the number of bytes written
// in the binary string, which will be fewer bytes than the original
// string if it had hex characters.
int CopyPrintableBufferToVector (const char* pb, size_t pb_bytes,
				  char* v, size_t v_bytes)
{
  char hex[] = "0123456789abcdef";
  char singlebyte[] = "\n\\\r\t\'\"";
  char       code[] = "n\\rt'\"";

  int int_bytes = pb_bytes;
  int jj=0;
  for (int ii=0; ii<int_bytes;) {
    if (jj>=int(v_bytes)) 
      throw string("Not enough room in CopyPrintableBufferToVector");
    if (pb[ii]=='\\' && ii+1<int_bytes) { // non-printable, so was escaped
      char* where = strchr(code, pb[ii+1]);
      if (where) {
	v[jj++] = singlebyte[where-code]; 
	ii+=2;
	continue;
      } else if (ii+3<int_bytes && pb[ii+1]=='x') { 
	v[jj++] = 
	  (strchr(hex, pb[ii+2])-hex)*16 + 
	  (strchr(hex, pb[ii+3])-hex);
	ii+=4;
	continue;
      } else {
	throw string("Malformed Numeric vector string:"+
		     string(pb, int_bytes)+" ... Error happened at:");
	// IntToString(ii));
      }
      
    } else { // printable, so take as is
      v[jj++] = pb[ii++];
    }
  }
  return jj;
}

// Helper Function for Pickling:  Turns a binary string into an
// ASCII printable string, where all the escape codes are spelled
// out:  i.e., single char hex value ff becomes 4 characters: \xff.
// The final length of the new string is reflected in the Array.
void PrintBufferToString (const char* vbuff, int bytes, Array<char>& a)
{
  char singlebyte[] = "\n\\\r\t\'\"";
  char       code[] = "n\\rt'\"";
  char        hex[] = "0123456789abcdef";
 
  for (int ii=0; ii<bytes; ii++) {
    unsigned char c = vbuff[ii];
    char* where = strchr(singlebyte, c);
    if (c!='\0' && where) {  // Awkward escape sequence?
      a.append('\\');
      a.append(code[where-singlebyte]);
    } else if (isprint(c)) { // Is printable? As as you are
      a.append(c); 
    } else { // Not strange, not printable, hex escape sequence it!
      a.append('\\'); a.append('x');
      int high_nibble = c >> 4; 
      char msn = hex[high_nibble]; 
      a.append(msn);
      
      int low_nibble = c & 0x0F; 
      char lsn = hex[low_nibble];
      a.append(lsn);
    }
  }
  a.append('\0'); // Since this is a C-Ctyle printable string
}

// Distinguish between M2k and OpenContainers usage
#if defined(OC_USE_OC) || defined(OC_NEW_STYLE_INCLUDES)
PTOOLS_END_NAMESPACE
#endif
