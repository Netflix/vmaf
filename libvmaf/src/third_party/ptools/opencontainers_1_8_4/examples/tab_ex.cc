
#include "ocval.h"
#include "ocserialize.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

int main()
{
  Val internal;
  internal.test();  // This should be a NO-OP if everything compiles okay

  // Insert all different types into the table
  Tab t;
  t["a"] = "100";
  t["b"] = 1;
  t["c"] = 3.0;
  t["truth"] = bool(1==1);
  cerr << t << endl;

  // See if table contains: Notice the Tab has the same interface as
  // any of the Associateive containers in the OpenContainers.
  if (t.contains("a")) {
    cerr << "yes" << endl;
  } else {
    cerr << "no" << endl;
  }

  // Insert directly: Like sub["100"] = 300, but without the default
  // constructor for Val (which isn't actually that expensive).
  Tab sub;
  sub.insertKeyAndValue("100", 300);

  // Note:  Tables can contain tables!
  t["sub"] = sub;

  // Cascading lookups are easy ... careful, if you try to
  // defereference a non-Tab, you get a logic_error!
  cerr << t["sub"]["100"]  << endl;

  t["sub"]["100"] = 500;

  // Getting a value out: Note that it out converts to the type you
  // request.
  real_8 a = t["c"];
  cerr << "a is " << a << endl;
  
  // Print out a table: Note that they look like Python tables (on
  // purpose!).
  cerr << t << endl;

  // This routine recursively descends and counts the total number of
  // elements in ALL nested subtables.
  cerr << "total_elements = " << t.total_elements() << endl;

  {
    // Note that ANY val can cast out as a string.
    Tab tt;
    tt["hello"] = 1.2646e6;
    real_8 dd = tt["hello"];
    Str s = tt["hello"];
    cerr << "s = " << s << " dd = " << dd << endl;
  }


  // We use Str because it seems to be twice as fast
  // for "human readable" strings, but if you are using
  // STL::string, you need those too ...
  Val v("hello");
  Str sss = v;
  string s = sss.c_str(); // How to get to a "string"

  Val zero(0);  // What is 0?  NULL?  an int?  A char *?
  cerr << "zero is " << zero << " with tag:" << zero.tag << endl;

  /*  This may not compile because NULL is an odd duck
  Val nullify(NULL); // What is a NULL?
  cerr << "nullify is " << nullify << " with tag:" << nullify.tag << endl;
  */

  // Built a real big table
  Tab ss;
  for (int ii=0; ii<1000; ii++) {
    Val in(ii); Str s=in; s+="12345678910";
    ss[s] = ii;
  }
  // Show how to iterate through tables and lookup values
  for (It ll(ss); ll(); ) {
    cerr << "key = " << ll.key() << " value = " << ll.value() << endl;
  }
  
  t["ssgs"] = ss;

  // Serialization/Deserialization.  Reasonably fast.  Note that we
  // have to count how many bytes to serialize first before we
  // allocate the buffer to serialize to.
  size_t bytes = BytesToSerialize(t);
  
  char* mem = new char[bytes];
  char* other = new char[bytes];
  Val res;
  for (int kk=0; kk<5e3; kk++) {
    Serialize(t, mem);
    memcpy(other, mem, bytes);
    res = Val();
    Deserialize(res, other);
    memcpy(other, mem, bytes);
  }
  if (memcmp(other, mem, bytes)!=0) 
    cerr << "Error!  Serialization doesn't work!" << endl;
  
  delete [] mem;
  delete [] other;

  Tab new_tab = "{ 0:0, 1:1, 2:2, 3:3, 4:'hello', 5:{'key':3.1}}";
  Tab add     = "{ 10:10, 11:11 }";
  cout << new_tab << endl;
  new_tab += add;
  cout << new_tab << endl;
  Tab add_to = "{ 10: 10000, 11:'yo' }";
  new_tab += add_to;
  cout << new_tab << endl;


  Tab next;
  next.append(111);
  next.append(222);
  next.prettyPrint(cout);
  next.appendStr(333);
  next.prettyPrint(cout);

  Tab n2;
  n2.appendStr(111);
  n2.appendStr(222);
  n2.prettyPrint(cout);
  n2.append(333);
  n2.prettyPrint(cout);

  return 0;
}
