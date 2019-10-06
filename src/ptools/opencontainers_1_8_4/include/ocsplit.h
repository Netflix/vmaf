#ifndef OCSPLIT_H_
#define OCSPLIT_H_

// Implement some standard Python string functions:
// Split, strip, Lower, Upper


#include "ocval.h"
#include <ctype.h>

PTOOLS_BEGIN_NAMESPACE 

// Inplace Upper
inline void Upper (string& inplace)
{
  const int len = inplace.length();
  char* data = &inplace[0];
  for (int ii=0; ii<len; ii++) {
    data[ii] = toupper(data[ii]);
  }
}

// Out-of-place Upper
inline string Upper (const string& out_of_place)
{
  string copy(out_of_place);
  Upper(copy);
  return copy;
}

// Inplace Lower
inline void Lower (string& inplace)
{
  const int len = inplace.length();
  char* data = &inplace[0];
  for (int ii=0; ii<len; ii++) {
    data[ii] = tolower(data[ii]);
  }
}

// Out-of-place Lower
inline string Lower (const string& out_of_place)
{
  string copy(out_of_place);
  Lower(copy);
  return copy;
}

// Return a list of the words in the string S, using sep as the
// delimiter string.  If maxsplit is given, at most maxsplit
// splits are done. If sep is not specified or is None, any
// whitespace string is a separator and empty strings are removed
// from the result.
inline Array<string> Split (const string& string_to_split,
			    const Val& sep=None, int max_split=-1)
{
  Array<string> result; // RVO
  const int len = string_to_split.length();
  const char* str = &string_to_split[0];
  int splits_so_far = 0;

  // Special case for maxsplit
  if (max_split==0) {
    result.append(string_to_split);
    return result;
  }
  
  // No sep specified, use whitespace, otherwise entire string
  // becomes the separator
  bool white_space_looking = true;
  string separator(" "); // NEEDS to be space for whitespacelooking!
  if (sep!=None) { // user-defined
    separator = string(sep);
    white_space_looking = false;
  }
  const char* separator_data = separator.data();
  const int separator_len    = separator.length();

  // Pattern match, looking for any in
  int next_start = 0;
  int ii = 0;
  bool end_of_str=false; // go one over for end of string processing
  while (!end_of_str) {

    // Last character processing, consider end of string as a final whitespace
    char c = ' ';
    bool match = false;
    if (ii<len) {
      
      // Still data coming: look for match
      c = str[ii];
      if (white_space_looking) {     // Whitespace 
	match = isspace(c);
      } else {                       // User-match
	if (separator[0]==c) {
	  // make sure still enough string left to compare against separator
	  int len_left = len-ii;
	  match = (separator_len<=len_left && 
		   (memcmp(separator_data, &str[ii], separator_len)==0));
	}
      }
    } else {
      // End of data, "implicit match" at end of string
      match = end_of_str = true;
    }

    // Match made, bring out string
    if (match) {
      const int next_string_len = ii-next_start;
      if (!(white_space_looking && next_string_len==0)) { 
	string next = string(str+next_start, next_string_len);
	result.append(next);
	splits_so_far += 1;      
      }
      next_start = ii+separator_len; 

      // Early return for splits reason
      if (max_split!=-1 && splits_so_far >= max_split) {
	if (next_start < len) {
	  result.append(string(str+next_start, len-next_start));
	}
	break;
      }

      ii += separator_len;
    } else {
      // Next char
      ii += 1;
    }

  }

  return result;
}



// Parse the URL into three components:
// http://www.yahoo.com/file/path
// protocol              http:
// host                  www.yahoo.com
// resource path         /file/path
// An runtime exception will be thrown if the url is malformed.
inline void ParseURL (const string& url,
		      string& protocol,string& host,string& path,string& port)
{
  Array<string> url_parse = Split(url, "://", 1);
  protocol = url_parse[0];
  Array<string> path_parse = Split(url_parse[1], "/", 1);
  host = path_parse[0];
  if (path_parse[1].find(":") != string::npos) { // port number?
    // split on port
    Array<string> possible_port = Split(path_parse[1], ":", 1);
    if (possible_port[0].find("/") == string::npos) {
      path = possible_port[0];
    } else {
      path = "/"+possible_port[0];
    }
    port = possible_port[1];
  } else {
    // no port number
    if (path_parse[1].find("/") == string::npos) {
      path = path_parse[1];
    } else {
      path = "/"+path_parse[1];
    }
    port = "";
  }
}

// Helper function for Split:
// Returns -1 if entire string contains elements of search_set,
// otherwise, the index at which the search fails
inline int SplitSearch_ (const char* string_to_search, 
			 int start_search, int times, int direction,
			 const char* search_set, int search_set_len)
{
  int search_index=start_search;
  for (int ii=0; ii<times; ii+=1, search_index+=direction) {
    char c = string_to_search[search_index];
    
    // See if the character of interest is in there
    int in_there = -1;
    for (int jj=0; jj<search_set_len; jj++) {
      if (c==search_set[jj]) {
	in_there = jj;
	break;
      }
    }
    // -1, nope, <>-1, index
    if (in_there != -1) { 
      continue; // yep, continue stripping
    } else {
      return search_index; 
    }
  }
  return -1;
}

//  Strip(string S, [chars]) -> string 
//    
//    Return a copy of the string S with leading and trailing
//    whitespace removed.
//    If chars is given and not None, remove characters in chars instead.
inline string Strip (const string& string_to_strip, const Val chars=None)
{
  string result;
  string set_of_chars_to_strip = " \n\r\t";
  if (chars!=None) {
    set_of_chars_to_strip = string(chars);
  } 
  const char* set_lookup = set_of_chars_to_strip.data();
  const int   set_lookup_len = set_of_chars_to_strip.length();
  const char* str = string_to_strip.data();
  const int str_len = string_to_strip.length();
  
  int front_end = SplitSearch_(str, 0, str_len, 1, 
			       set_lookup, set_lookup_len);
  if (front_end != -1) {
    int back_end = SplitSearch_(str, str_len-1 ,str_len, -1,
				set_lookup,set_lookup_len);
    if (front_end<=back_end) {
      result = string(&string_to_strip[front_end], back_end-front_end+1);
    }
  }
  return result;
}


PTOOLS_END_NAMESPACE

#endif // OCSPLIT_H_
