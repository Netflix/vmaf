
// This timing test compares 4 different "comparable" implementations:
// We compare the STL::map versus three "comparable" classes, The
// HashTableT, AVLTreeT and AVLHashT.

// ///////////////////////////////////////////// Defines

// Choose one of the implementations to try out and time
// #define IMPL_HASH
// #define IMPL_AVLTREE
// #define IMPL_AVLHASH
// #define IMPL_MAP

// ///////////////////////////////////////////// Include Files

#if defined(IMPL_MAP)
#  include <map>
   using std::map;
#endif

#if defined(HAVE_STL)
#  include <iostream>
#  include <string>
   using namespace std;

#  include "ochashfunction.h"
   inline unsigned long HashFunction (const string& s)
   {
    return OCStringHashFunction(s.data(), s.length());
   }
   

#else

#  include <stdio.h>
#  include "ocstring.h"
   typedef OCString string;

#endif


#include "ocport.h"
#include "ochashtablet.h"
#include "ocavltreet.h"
#include "ocavlhasht.h"



// ///////////////////////////////////////////// Globals 

// Generate a string and rotate it around
string shift (const string& s, int ii)
{
  string ret = s;
  const int len = s.length();
  for (int jj=0; jj<len; jj++) {
    ret[(jj+ii) % len] = s[jj];
  }
  char buf[1024]; 
  snprintf(buf, 1024, "%d", ii);
  ret += string(buf);
  return ret;
}


// ///////////////////////////////////////////// MyTime Class 

#include <sys/time.h>
// A bit of a hack for getting times on a UNIX system
class MyTime {
  public:
    MyTime () 
    {
      if (gettimeofday(&timeval_, NULL) <0) 
	// throw LogicException("system time()");
	cerr << "system time()" << endl;
    }
    operator double()  
    { 
      double usec = timeval_.tv_usec;
      double sec  = timeval_.tv_sec;
      // cerr << usec << " " << sec << endl;
      // return sec + (usec/1e-6);
      return sec + usec / 1e6;
    }
  private:
    struct timeval timeval_;
    
}; // MyTime



// ///////////////////////////////////////////// HashTest Methods

int main ()
{
  int numb = 1000; // e1;

  {
    for (int len = 1; len<5e3; len = len*2) {
      
#if defined(IMPL_HASH)
      HashTableT<string,string,8> impl; 
#elif defined(IMPL_AVLTREE)
      AVLTreeT<string,string,8> impl;
#elif defined(IMPL_AVLHASH)
      AVLHashT<string,string,8> impl;
#elif defined(IMPL_MAP)
      map<string,string> impl;
#endif

      string key = "The quick brown fox jumped over the lazy dogs";
      string value = "The quick brown fox jumped over the lazy dogs many many times";
      string* keys = new string[len];
      string* values = new string[len];
      for (int ii=0; ii<len; ii++) {
	keys[ii] = shift(key, ii);
	values[ii] = shift(value, ii);
      }
      MyTime start;
      for (int jj=0; jj<numb; jj++) {
	int ii;
	for (ii=0; ii<len; ii++) 
#if defined(IMPL_MAP)
	  impl.insert(map<string,string>::value_type(keys[ii], values[ii])); 
#else 
	  impl.insertKeyAndValue(keys[ii], values[ii]);
#endif
	for (ii=0; ii<len; ii++) 
#if defined(IMPL_MAP)
	  impl.erase(keys[ii]); 
#else
	  impl.remove(keys[ii]);
#endif
      }
      MyTime end;
      cerr << "Time for " << len << " i/d (done " << numb << " times) = " << end - start << endl;
      delete [] keys; 
      delete [] values;
    }
  }
 

  {
    for (int len = 1; len<1e4; len = len*2) {
#if defined(IMPL_HASH)
      HashTableT<string,string,8> impl; 
#elif defined(IMPL_AVLTREE)
      AVLTreeT<string,string,8> impl;
#elif defined(IMPL_AVLHASH)
      AVLHashT<string,string,8> impl;
#elif defined(IMPL_MAP)
      map<string,string> impl;
#endif

      string key = "The quick brown fox jumped over the lazy dogs";
      string value = "The quick brown fox jumped over the lazy dogs many many times";
      string* keys = new string[len];
      string* values = new string[len];
      int ii;
      for (ii=0; ii<len; ii++) {
	keys[ii] = shift(key, ii);
	values[ii] = shift(value, ii);
      }
      
      for (ii=0; ii<len; ii++) {
#if defined(IMPL_MAP)
	impl.insert(map<string,string>::value_type(keys[ii], values[ii]));
#else
	impl.insertKeyAndValue(keys[ii], values[ii]);
#endif
      }
      MyTime start;
      for (int jj=0; jj<numb; jj++) {
	for (int kk=0; kk<len; kk++) {
	  string& s = impl[keys[kk]];
	}
      }
      MyTime end;
      for (ii=0; ii<len; ii++) {
#if defined(IMPL_MAP)
	impl.erase(keys[ii]);
#else 
	impl.remove(keys[ii]);
#endif
      }
      cerr << "Time for " << len << " lookups (done " << numb << " times) = " << end - start << endl;
      delete [] keys; 
      delete [] values;
    }
  }
  
}





