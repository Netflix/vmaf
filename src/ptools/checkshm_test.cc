
#include "checkshm.h"  


int lookup (StreamingPool *sp, size_t bytes, bool how)
{
  Arr things = "[None, 'a', 1, 1.1, [], {}, array([], 'i')]";
  cout << "** Checking: " << things << endl;
  for (size_t ii=0; ii<things.length(); ii++) {
    try {
      cout << InSHM(things[ii], sp, how, "top-level") << " : " << things[ii] 
	   << endl;
    } catch (const exception& e) {
      cout << e.what() << endl;
    }
  }

  {
    Val thing2 = Shared(sp, things);
    Arr& a = thing2;
    cout << "** Checking: " << thing2 << endl;
    for (size_t ii=0; ii<a.length(); ii++) {
      try {
	cout << InSHM(a[ii], sp, how, "top-level") << " : " << a[ii] 
	     << endl;
      } catch (const exception& e) {
	cout << e.what() << endl;
      }
    }
    
    cout << "Replacing one at a time" << endl;
    for (size_t jj=0; jj<a.length(); jj++) {
      Val s = things[jj];
      s.swap(a[jj]);     
      
      for (size_t ii=0; ii<a.length(); ii++) {
	try {
	  cout << InSHM(a[ii], sp, how, "top-level") << " : " << a[ii] 
	       << endl;
	} catch (const exception& e) {
	  cout << e.what() << endl;
	}
      }
    }
  }

  {
    // Put a tab and some nested Tabs in memory and make sure they
    // are all created there
    Val v = 
      Shared(sp, Tab("{'a':1, 'b':2.2, 'c':'three', 'd':[], 'e':{'w':1}}"));
    try {
      cout << InSHM(v, sp, how, "top-level") << " : " << v << endl;
    } catch (exception& e) {
      cout << e.what() << endl;
    }
    for (It ii(v); ii(); ) {
      const Val& key = ii.key();
      Val& value = ii.value();
      try {
	cout << InSHM(key, sp, how, "top-level") << " : " << key << endl;
      } catch (exception& e) {
	cout << e.what() << endl;
      }
      try {
	cout << InSHM(value, sp, how, "top-level") << " : " << value << endl;
      } catch (exception& e) {
	cout << e.what() << endl;
      }
    }
    v["new entry1"] = Tab();
    cout << v << endl;
    try {
      cout << " " << InSHM(v["new entry1"], sp, how, "top-level") << " : " << v["new entry1"] << endl;
    } catch (exception& e) {
      cout << e.what() << endl;
    }
    v["new entry2"] = Tab("{'a':{ 'b': 1 } }");
    cout << v << endl;
    try {
      cout << " " << InSHM(v["new entry2"], sp, how, "top-level") << " : " << v["new entry2"] << endl;
    } catch (exception& e) {
      cout  << e.what() << endl;
    }
  }

  {
    // Put a tab and some nested Tabs in memory and make sure they
    // are all created there
    Val v = Tab("{'a':1, 'b':2.2, 'c':'three', 'd':[], 'e':{'w':1}}");
    try {
      cout << InSHM(v, sp, how, "top-level") << " : " << v << endl;
    } catch (exception& e) {
      cout << e.what() << endl;
    }
    for (It ii(v); ii(); ) {
      const Val& key = ii.key();
      Val& value = ii.value();
      try {
	cout << InSHM(key, sp, how, "top-level") << " : " << key << endl;
      } catch (exception& e) {
	cout << e.what() << endl;
      }
      try {
	cout << InSHM(value, sp, how, "top-level") << " : " << value << endl;
      } catch (exception& e) {
	cout << e.what() << endl;
      }
    }
    v["new entry1"] = Tab();
    cout << v << endl;
    try {
      cout << " " << InSHM(v["new entry1"], sp, how, "top-level") << " : " << v["new entry1"] << endl;
    } catch (exception& e) {
      cout << e.what() << endl;
    }
    v["new entry2"] = Tab("{'a':{ 'b': 1 } }");
    cout << v << endl;
    try {
      cout << " " << InSHM(v["new entry2"], sp, how, "top-level") << " : " << v["new entry2"] << endl;
    } catch (exception& e) {
      cout  << e.what() << endl;
    }
  }
  
  

  return 0;
}


int main (int argc, char**argv)
{
  size_t bytes = 1000000;
  char* mem = new char[bytes];
  StreamingPool* sp = StreamingPool::CreateStreamingPool(mem, bytes, 8);
  

  lookup(sp, bytes, false);  // just return true or false
  lookup(sp, bytes, true);   // throw exception with more info
}
