#ifndef URLENCODE_H_
#define URLENCODE_H_

// URLEncode a Val for shipping across HTTP sockets

#include "ocsplit.h"

// Take a string an make it url safe: all funky characters
// get translated, and we keep numbers and alphanumerics.
string URLEncode (const char *s, int len=-1)
{
  static const char hexdigits[]="0123456789abcdef";
  string res;
  if (len==-1) len=strlen(s);
  for (int ii=0; ii<len; ii++) {
    char c = s[ii];
    if (isalnum(c)) {
      res.append(1, c);
    } else if (c==' ') {
      res.append(1, '+');
    } else {
      signed int cc = c;
      char hexstr[4] = { '%',0 };
      hexstr[1] = hexdigits[cc/16];
      hexstr[2] = hexdigits[cc%16];
      res.append(string(hexstr,3));
    }
  }
  return res;  
}
string URLEncode (const string& s)
{ return URLEncode(s.data(), s.length()); }

// helper function to handle tables for us
template <class T> 
string URLEncodeTable_ (const T& o)
{
  string result;
  int len = o.entries();
  It ii(o); 
  for (int jj=0; ii(); jj++) {
    string key = ii.key();
    string value = ii.value();

    string keyvalue = URLEncode(key)+ "=" + URLEncode(value);    
    if (jj!=len-1) keyvalue += "&"; 

    result.append(keyvalue);
  }
  return result;
}

// Table: get all key-values
string URLEncode (const OTab& o)
{ return URLEncodeTable_(o); }
string URLEncode (const Tab& o)
{ return URLEncodeTable_(o); }

// helper function for arrays
template <class T>
string URLEncode (const Array<T>& a)
{
  string res;
  string kv;
  int len = a.length();
  for (int ii=0; ii<len; ii++) {
    string s = Val(a[ii]); // gets the right stringize
    if (ii==len-1) {
      kv = Stringize(ii)+"="+s;
    } else {
      kv = Stringize(ii)+"="+s+"&";
    }
  }
  return res;
}

#define URLENCODEARR(T) { Array<T>& a=v; return URLEncode(a); }
string URLEncodeArray_ (const Val& v)
{
  switch (v.subtype) {
  case 's': URLENCODEARR(int_1); 
  case 'S': URLENCODEARR(int_u1); 
  case 'i': URLENCODEARR(int_2); 
  case 'I': URLENCODEARR(int_u2); 
  case 'l': URLENCODEARR(int_4); 
  case 'L': URLENCODEARR(int_u4); 
  case 'x': URLENCODEARR(int_8); 
  case 'X': URLENCODEARR(int_u8); 
  case 'f': URLENCODEARR(real_4); 
  case 'F': URLENCODEARR(complex_8); 
  case 'd': URLENCODEARR(real_8); 
  case 'D': URLENCODEARR(complex_16);
  case 'b': URLENCODEARR(bool);
  case 'Z': URLENCODEARR(Val);
  case 'a': URLENCODEARR(string);
  default: throw runtime_error("Can only URLEncode arrays of POD");
  }
}

string URLEncode (const Tup& a) { return URLEncode(a.impl()); }

// Dispatch to proper url encoder
string URLEncode (const Val& v)
{
  if (v.tag=='o') {
    OTab& o = v;
    return URLEncode(o);
  } else if (v.tag=='t') {
    Tab& o = v;
    return URLEncode(o);
  } else if (v.tag=='u') {
    Tup& u = v;
    return URLEncode(u);
  } else if (v.tag=='n') {
    return URLEncodeArray_(v);
  } else {
    string s = v;
    return URLEncode(s);
  }
}

#endif // URLENCODE_H_
