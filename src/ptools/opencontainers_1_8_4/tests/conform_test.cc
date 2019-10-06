
#include "ocport.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

#include "occonforms.h"


void mesg (bool expected, const char* preamble,
	   const Val& instance, const Val& prototype, 
	   bool structure_match=true, Conform_e type_match=EXACT_MATCH,
	   bool thrw=false)
{
  // strings for header
  string sm = structure_match ? "YES" : "NO";
  string conform;
  if (type_match==EXACT_MATCH) 
    conform = "EXACT_MATCH";
  else if (type_match==LOOSE_MATCH) 
    conform= "LOOSE_MATCH";
  else if (type_match==LOOSE_STRING_MATCH)  
    conform = "LOOSE_STRING_MATCH";
  else 
    throw runtime_error("Illegal conform?"+Stringize(int(type_match)));

  
  // info about test
  cout << "---------" << preamble << "----- structure_match:" << sm << " type_match:"<< conform << " " << thrw << endl;
  cout << "..............instance............" << endl;
  instance.prettyPrint(cout);
  cout << endl;
  cout << "..............prototype............." << endl;
  prototype.prettyPrint(cout);
  cout << endl;

  // perform conforms
  bool result = false;
  bool threw = false;
  try {
    result = Conforms(instance, prototype, structure_match, type_match, thrw);
  } catch (const runtime_error& e) {
    threw = true;
    cout << "**MESSAGE CAUGHT:" << endl << e.what() << endl;
  }


  // result
  cout << "--> result: " << ((result) ? "true" : "false") << endl; 
  if (result!=expected) {
    cerr << "!!!!!!!!!!!!!!! NOT EXPECTED !!!!!!!!!!!" << endl;
    exit(1);
  }
  if (result==false && thrw==true && threw==false) {
    cerr << "!!!!!!!!!!!!!!! No error message thrown??? !!!!!!!!" << endl;
    exit(1);
  }
}


// should always be true, under all variations
void easyCheck (const Val& instance, const Val& prototype, bool thr)
{
  static bool      keymatch[] = { true, false };
  static Conform_e trials[] = { EXACT_MATCH, LOOSE_MATCH, LOOSE_STRING_MATCH };
  {
    for (size_t ii=0; ii<sizeof(keymatch)/sizeof(bool); ii++) {
      for (size_t jj=0; jj<sizeof(trials)/sizeof(Conform_e); jj++) {
	mesg(true, "easy test", instance, prototype, 
	     keymatch[ii], trials[jj], thr);
	
      }
    }

  }
}

void failCheck (const Val& instance, const Val& prototype, bool thr)
{
  if (instance.tag == prototype.tag) return; // really just testing failure,
  // so we just bypass when tag is the same

  static bool      keymatch[] = { true, false };
  static Conform_e trials[] = { EXACT_MATCH, LOOSE_MATCH, LOOSE_STRING_MATCH };
  {
    for (size_t ii=0; ii<sizeof(keymatch)/sizeof(bool); ii++) {
      for (size_t jj=0; jj<sizeof(trials)/sizeof(Conform_e); jj++) {

	bool same_structure = keymatch[ii];
	Conform_e type_match = trials[jj];
	bool expected=false;

	// Exceptional cases
	if (instance==Eval("o{}") && prototype==Eval("{}") &&  
	    (type_match!=EXACT_MATCH)) expected=true;
	if (instance=="" && type_match==LOOSE_STRING_MATCH) expected=true;

	mesg(expected, "fail test", instance, prototype, 
	    same_structure, type_match, thr);
	
      }
    }

  }
}


int testing(bool thr)
{
  // Start simple
  {
    Val instance = Eval("{'a':1, 'b':2.2, 'c':'three'}");
    Val prototype= Eval("{'a':0, 'b':0.0, 'c':''}");
    mesg(true, "simple test", instance, prototype, 
	 true, EXACT_MATCH, thr);
  }

  {
    Val instance = Eval("{'a':1, 'b':2.2, 'c':'three'}");
    Val prototype= Eval("{'a':0, 'b':0.0}");
    mesg(false, "simple test, too many keys", instance, prototype,
	 true, EXACT_MATCH, thr);
  }

  {
    Val instance = Eval("{'a':1, 'b':2.2, 'c':'three'}");
    Val prototype= Eval("{'a':0, 'b':0.0}");
    mesg(true, "simple test, prototype a subset of keys", instance, prototype, 
	 false, EXACT_MATCH, thr);
  }

  {
    Val instance = Eval("{'a':1, 'b':2.2, 'c':'three'}");
    Val prototype= Eval("{'a':None, 'b':None}");
    mesg(true, "simple test, no type match", instance, prototype, 
	 false, EXACT_MATCH, thr);
  }

  {
    Val instance = Eval("{'a':1, 'b':2.2, 'c':'three'}");
    instance["a"] = int_u8(100);
    instance["b"] = real_4(3.141592658);
    Val prototype= Eval("{'a':0, 'b':0.0}");
    mesg(true, "simple test, no type match", instance, prototype, 
	 false, LOOSE_MATCH, thr);
  }

  {
    cout << "HERE" << endl;
    Val instance = Eval("{'a':1, 'b':2.2, 'c':'three'}");
    instance["a"] = int_u8(100);
    instance["b"] = real_4(3.141592658);
    Val prototype= Eval("{'a':0, 'b':0.0}");
    mesg(false, "simple test, no type match", instance, prototype, 
	 false, EXACT_MATCH, thr);
  }

  {
    Val instance = Eval("{'a':1, 'b':2.2, 'c':'three'}");
    instance["a"] = int_u8(100);
    instance["b"] = real_4(3.141592658);
    Val prototype= Eval("{'a':0, 'b':0.0}");
    mesg(false, "simple test, no type match", instance, prototype, 
	 false, EXACT_MATCH, thr);
  }

  {
    Val instance = Eval("{'a':1, 'b':2.2, 'c':'three'}");
    instance["a"] = int_u8(100);
    instance["b"] = real_4(3.141592658);
    Val prototype= Eval("{'a':0, 'b':0.0, 'c':0}");
    mesg(false, "simple test, no type match", instance, prototype, 
	 false, LOOSE_MATCH, thr);
  }

  {
    Val instance = Eval("{'a':1, 'b':2.2, 'c':'three'}");
    instance["a"] = int_u8(100);
    instance["b"] = real_4(3.141592658);
    Val prototype= Eval("{'a':0, 'b':0.0, 'c':0}");
    mesg(true, "simple test, no type match", instance, prototype, 
	 false, LOOSE_STRING_MATCH, thr);
  }

  {
    Val instance = Eval("[1,2.2, 'three']");
    Val prototype= Eval("[0,0.0, '']");
    mesg(true, "arr test, no type match", instance, prototype, 
	 true, EXACT_MATCH, thr);
  }

  {
    Val instance = Eval("[1,2.2, 'three']");
    Val prototype= Eval("[0,0.0, '']");
    mesg(true, "arr test, no type match", instance, prototype, 
	 false, EXACT_MATCH, thr);
  }


  {
    Val instance = Eval("[1,2.2, 'three']");
    Val prototype= Eval("[0,0.0, '']");
    mesg(true, "arr test, no type match", instance, prototype, 
	 true, LOOSE_MATCH, thr);
  }

  {
    Val instance = Eval("[1,2.2, 'three']");
    Val prototype= Eval("[0,0.0, '']");
    mesg(true, "arr test, no type match", instance, prototype, 
	 true, LOOSE_STRING_MATCH, thr);
  }


  {
    Val instance = Eval("[1,2.2, 'three']");
    Val prototype= Eval("[0,0.0, '']");
    mesg(true, "arr test, no type match", instance, prototype, 
	 false, LOOSE_MATCH, thr);
  }

  {
    Val instance = Eval("[1,2.2, 'three']");
    Val prototype= Eval("[0,0.0, '']");
    mesg(true, "arr test, no type match", instance, prototype, 
	 true, LOOSE_MATCH, thr);
  }

  {
    Val instance = Eval("[1, 2.2, 'three']");
    Val prototype= Eval("[0, 0.0 ]");
    mesg(false, "arr test, no type match", instance, prototype, 
	 true, EXACT_MATCH, thr);
  }

  {
    Val instance = Eval("[1, 2.2, 'three']");
    Val prototype= Eval("[0, 0.0 ]");
    mesg(true, "arr test, no type match", instance, prototype, 
	 false, EXACT_MATCH, thr);
  }

  {
    Val instance = Eval("[1, 2.2, 'three']");
    Val prototype= Eval("[0, 0.0 ]");
    mesg(true, "arr test, no type match", instance, prototype, 
	 false, LOOSE_MATCH, thr);
  }

  {
    Val instance = Eval("[1, 2.2, 'three']");
    instance[0] = int_u8(1);
    instance[1] = real_4(2.333333);
    Val prototype= Eval("[0, 0.0 ]");
    mesg(false, "arr test, no type match", instance, prototype, 
	 false, EXACT_MATCH, thr);
  }


  {
    Val instance = Eval("[1, 2.2, 'three']");
    instance[0] = int_u8(1);
    instance[1] = real_4(2.333333);
    Val prototype= Eval("[0, 0.0 ]");
    mesg(true, "arr test, no type match", instance, prototype, 
	 false, LOOSE_MATCH, thr);
  }


  {
    Val instance = Eval("[1, 2.2]");
    Val prototype= Eval("[0, 0.0, '' ]");
    mesg(false, "arr test, no type match", instance, prototype, 
	 false, LOOSE_MATCH, thr);
  }

  {
    Val instance = Eval("['1', '2.2', 'three']");
    Val prototype= Eval("[0, 0.0, '' ]");
    mesg(true, "arr test, no type match", instance, prototype, 
	 false, LOOSE_STRING_MATCH, thr);
  }

    
  easyCheck(1.0, None, thr);
  easyCheck(1.0f, None, thr);
  easyCheck(int_1(1), None, thr);
  easyCheck(int_2(1), None, thr);
  easyCheck(int_4(1), None, thr);
  easyCheck(int_8(1), None, thr);
  easyCheck(int_u1(1), None, thr);
  easyCheck(int_u2(1), None, thr);
  easyCheck(int_u4(1), None, thr);
  easyCheck(int_u8(1), None, thr);
  easyCheck(complex_8(1), None, thr);
  easyCheck(complex_16(1), None, thr);
  easyCheck(int_n(1), None, thr);
  easyCheck(int_un(1), None, thr);
  easyCheck(Tab(), None, thr);
  easyCheck(OTab(), None, thr);
  easyCheck(Arr(), None, thr);
  easyCheck(Tup(), None, thr);
  easyCheck(Array<int_1>(), None, thr);

  easyCheck(1.0, 0.0, thr);
  easyCheck(1.0f, 0.0f, thr);
  easyCheck(int_1(1), int_1(0), thr);
  easyCheck(int_2(1), int_2(0), thr);
  easyCheck(int_4(1), int_4(0), thr);
  easyCheck(int_8(1), int_8(0), thr);
  easyCheck(int_u1(1), int_u1(0), thr);
  easyCheck(int_u2(1), int_u2(0), thr);
  easyCheck(int_u4(1), int_u4(0), thr);
  easyCheck(int_u8(1), int_u8(0), thr);
  easyCheck(complex_8(1), complex_8(0), thr);
  easyCheck(complex_16(1), complex_16(0), thr);
  easyCheck(int_n(1), int_n(0), thr);
  easyCheck(int_un(1), int_un(0), thr);
  easyCheck(Tab(), Tab(), thr);
  easyCheck(OTab(), OTab(), thr);
  easyCheck(Arr(), Arr(), thr);
  easyCheck(Tup(), Tup(), thr);
  easyCheck(Array<int_1>(), Array<int_1>(), thr);
  {
    Val instance = Eval("{'a':1, 'b':2.2, 'c':'three', 'd':[1,2.2,3.3],"
			" 'nested':{'a':1, 'b':2.2, 'c':'three', 'd':[1,{},[]]}"
			"}");
    Val prototype = Eval("{'a':1, 'b':2.2, 'c':'three', 'd':[1,2.2,3.3],"
			" 'nested':{'a':1, 'b':2.2, 'c':'three', 'd':[1,{},[]]}"
			"}");
    easyCheck(instance, prototype, thr);
  }

  Arr fails = "['', {}, o{}, 1, 1.0, (1+1j)]";
  for (size_t ii=0; ii<fails.length(); ii++) {
    const Val& failval = fails[ii];
    if (real_8(failval)!=1.0) { 
      failCheck(1.0, failval, thr);
      failCheck(1.0f, failval, thr);
    }
    if (int(failval)!=1) { 
      failCheck(int_1(1), failval, thr);
      failCheck(int_2(1), failval, thr);
      failCheck(int_4(1), failval, thr);
      failCheck(int_8(1), failval, thr);
      failCheck(int_u1(1), failval, thr);
      failCheck(int_u2(1), failval, thr);
      failCheck(int_u4(1), failval, thr);
      failCheck(int_u8(1), failval, thr);
    }
    if (failval.tag!='D') {
      failCheck(complex_8(1), failval, thr);
      failCheck(complex_16(1), failval, thr);
    }
    failCheck(int_n(1), failval, thr);
    failCheck(int_un(1), failval, thr);
    failCheck(Tab(), failval, thr);
    failCheck(OTab(), failval, thr);
    failCheck(Arr(), failval, thr);
    failCheck(Tup(), failval, thr);
    failCheck(Array<int_1>(), failval, thr);
    failCheck("", failval, thr);
  }

  {
    Val instance = Eval("{'a':1, 'b':2.2, 'c':'three', 'd':[1,2.2,3.3],"
			" 'nested':{'a':1, 'b':2.2, 'c':'three', 'd':[1,{},[]]}"
			"}");
    Val prototype = Eval("{'a':1, 'b':2.2, 'c':'three', 'd':[1,2.2],"
			"'nested':o{'a':1, 'b':2.2, 'c':'three', 'd':[1,{},[]]}"
			 "}");

    mesg(false, "arr test, no type match", instance, prototype, 
	 true, EXACT_MATCH, thr);
    mesg(false, "arr test, no type match", instance, prototype, 
	 false, EXACT_MATCH, thr);
    mesg(false, "arr test, no type match", instance, prototype, 
    	 true, LOOSE_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 false, LOOSE_MATCH, thr);
    mesg(false, "arr test, no type match", instance, prototype, 
    	 true, LOOSE_STRING_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 false, LOOSE_STRING_MATCH, thr);
  }

  {
    Val instance = Eval("o{'a':1, 'b':2.2, 'c':'three', 'd':(1,2.2,3.3),"
			" 'nested':{'a':1, 'b':2.2, 'c':'three', 'd':[1,{},[]]}"
			"}");
    Val prototype = Eval("{'a':1, 'b':2.2, 'c':'three', 'd':[1,2.2],"
			"'nested':o{'a':1, 'b':None,'c':'three', 'd':[1,{},[]]}"
			 "}");

    mesg(false, "arr test, no type match", instance, prototype, 
	 true, EXACT_MATCH, thr);
    mesg(false, "arr test, no type match", instance, prototype, 
	 false, EXACT_MATCH, thr);
    mesg(false, "arr test, no type match", instance, prototype, 
    	 true, LOOSE_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 false, LOOSE_MATCH, thr);
    mesg(false, "arr test, no type match", instance, prototype, 
    	 true, LOOSE_STRING_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 false, LOOSE_STRING_MATCH, thr);
  }

  {
    Val instance = Eval("array([1,2,3], 'i')");
    Val prototype = Eval("array([1,2,3], 'i')");

    mesg(true, "arr test, no type match", instance, prototype, 
	 true, EXACT_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
	 false, EXACT_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 true, LOOSE_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 false, LOOSE_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 true, LOOSE_STRING_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 false, LOOSE_STRING_MATCH, thr);
  }

  {
    Val instance = Eval("array([1,2,3], 'f')");
    Val prototype = Eval("array([1,2,3], 'd')");
    mesg(false, "arr test, no type match", instance, prototype, 
	 true, EXACT_MATCH, thr);
    mesg(false, "arr test, no type match", instance, prototype, 
	 false, EXACT_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 true, LOOSE_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 false, LOOSE_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 true, LOOSE_STRING_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 false, LOOSE_STRING_MATCH, thr);
  }

  {
    Val instance = Eval("array([1,2,3], 'f')");
    Val prototype = Eval("[1,2,3]");
    mesg(false, "arr test, no type match", instance, prototype, 
	 true, EXACT_MATCH, thr);
    mesg(false, "arr test, no type match", instance, prototype, 
	 false, EXACT_MATCH, thr);
    mesg(false, "arr test, no type match", instance, prototype, 
    	 true, LOOSE_MATCH, thr);
    mesg(false, "arr test, no type match", instance, prototype, 
    	 false, LOOSE_MATCH, thr);
    mesg(false, "arr test, no type match", instance, prototype, 
    	 true, LOOSE_STRING_MATCH, thr);
    mesg(false, "arr test, no type match", instance, prototype, 
    	 false, LOOSE_STRING_MATCH, thr);
  }

  {
    Val instance = Eval("[1,2,3]");
    Val prototype = Eval("(1,2,3)");
    mesg(false, "arr test, no type match", instance, prototype, 
	 true, EXACT_MATCH, thr);
    mesg(false, "arr test, no type match", instance, prototype, 
	 false, EXACT_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 true, LOOSE_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 false, LOOSE_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 true, LOOSE_STRING_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 false, LOOSE_STRING_MATCH, thr);
  }

  {
    Val instance = Eval("(1,2,3)");
    Val prototype = Eval("[1,2,3]");
    mesg(false, "arr test, no type match", instance, prototype, 
	 true, EXACT_MATCH, thr);
    mesg(false, "arr test, no type match", instance, prototype, 
	 false, EXACT_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 true, LOOSE_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 false, LOOSE_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 true, LOOSE_STRING_MATCH, thr);
    mesg(true, "arr test, no type match", instance, prototype, 
    	 false, LOOSE_STRING_MATCH, thr);
  }
  

  return 0;
}

int main()
{

  testing(false); // make sure still works as expected with "just" return false
  testing(true);  // make sure throws error when problems with good message

  return 0;
}
