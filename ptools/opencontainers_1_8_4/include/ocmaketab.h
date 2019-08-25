#ifndef MAKETAB_H_

#include "occonvert.h"

OC_BEGIN_NAMESPACE

// Force a given Val to convert to some sort of Tab.
// 
// Change the given Val INPLACE to some corresponding Tab.
// If the Val is a Tab, this is a no-op: if the Val is
// anything else, it constructs a Tab with a single key
// 'VALUE' with the previous Val.  For example:
//   Val v = 1.0f;  // float
//   MakeTab(v);
//   v.prettyPrint(cout); // { 'VALUE': 1.0 }

// If the given Val is an Arr, then you may want to change the Arr into
// a Tab by inserting elements of Arr into a Tab with keys 0..n-1.
// That is handled quite well by the occonvert routines.

inline void MakeTabInPlace (Val& v, 
			    bool convert_arr=false,bool keys_as_strings=false)
{
  if (v.tag=='t') return;

  if (convert_arr && v.tag=='n' && v.subtype=='Z') {
    // Convert Arr to Tab
    ConvertArrToTab(v, false, keys_as_strings);
  } else {
    // Put in table as VALUE 
    Val temp;  
    v.swap(temp);  // To avoid excessive copying, copy the old value into
                   // a temp in constant time
    
    v = Tab();
    Val& new_side = v["VALUE"]; // Inserts none into table
    new_side.swap(temp);        // avoids deep-copy
  }
}


// As above, but returns a Tab directly rather than changing it inplace
inline Tab MakeTab (const Val& v, 
		    bool convert_arr=false, bool keys_as_strings=false)
{
  Tab ret_val;             // Enable return value optimization
  if (v.tag=='t') { 
    ret_val = v;           // Because returns copy, does a deep copy here
  } else if (convert_arr && v.tag=='n' && v.subtype=='Z') {
    Val vcopy(v);  // Deep copy 
    ConvertArrToTab(vcopy, false, keys_as_strings);
    Tab& t = vcopy;
    ret_val.swap(t);  // Once a Tab, quick copy into retval
  } else {
    ret_val["VALUE"] = v;  // Because returns copy, does a deep copy here
  }
  return ret_val;
}

OC_END_NAMESPACE


#define MAKETAB_H_
#endif // MAKETAB_H_
