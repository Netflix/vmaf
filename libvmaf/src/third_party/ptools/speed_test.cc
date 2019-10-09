
#include "chooseser.h"

// Test the speed of the old P2 loading vs. the new PickleLoader.  See
// results below.

void CreateBig (Val& v)
{
  Arr a  = "[ None, 1.0, 'hello', {}, {'a':1}, {'a':1, 'b':2}, [], [1], [1,2], [1,2,3]]";
  v = Tab();
  Tab& t = v;
  for (int ii=0; ii<10000; ii++) {
    if (ii%2==0) 
      t[Stringize(ii)] = a[ii%a.length()];
    else 
      t[ii] = a[ii%a.length()];
  }
  t["nested"] = t; // full copy
  for (int ii=0; ii<10000; ii++) {
    t["AAA"+Stringize(ii)] = "The quick brown fox jumped over the lazy dogs quite a few times on Sunday "+Stringize(ii);
  }
}

//#define CHECK_RESULTS() if (v!=result) cerr << "not same?" << endl;
#define CHECK_RESULTS()



// Results vary: some instances, the older stuff is quicker, in some
// instances the newer stuff is quicker.  Given that the newer stuff
// is significantly easier to maintain and update, the new stuff makes
// more sense.
int main (int argc, char**argv)
{
  if (argc!=2) {
    cerr << "Usage: " << argv[0] << " pickle0|unpickle0|unpickleold0|pickle2|unpickle2|unpickleold2|unpickleOC|unpickleM2k " << endl;
    exit(1);
  }
  Val v;
  CreateBig(v);

  Serialization_e ser=SERIALIZE_NONE;
  string which = argv[1];
  const int times = 200;
  string proto = "SERIALIZE_NONE";

  int direction = 1; // to pickle!
  if (which.find("pickleold0") != string::npos) {
    ser = SERIALIZE_P0_OLDIMPL;
    proto = "SERIALIZE_P0_OLDIMPL";
  }
  if (which.find("pickle0") != string::npos) {
    ser = SERIALIZE_P0;
    proto = "SERIALIZE_P0";
  }
  if (which.find("pickleold2") != string::npos) { 
    ser = SERIALIZE_P2_OLDIMPL;
    proto = "SERIALIZE_P2_OLDIMPL";
  }
  if (which.find("pickle2") != string::npos) { 
    ser = SERIALIZE_P2;
    proto = "SERIALIZE_P2";
  }
  if (which.find("pickleOC") != string::npos) {
    ser = SERIALIZE_OC;
    proto = "SERIALIZE_OC";
  }
  if (which.find("pickleM2K") != string::npos) {
    ser = SERIALIZE_M2K;
    proto = "SERIALIZE_M2K";
  }
  if (which.find("pickletext") != string::npos) { 
    ser = SERIALIZE_TEXT;
    proto = "SERIALIZE_TEXT";
  }
  if (which.find("picklepretty") != string::npos) { 
    ser = SERIALIZE_PRETTY;
    proto = "SERIALIZE_PRETTY";
  }
  if (which.find("un") != string::npos) {
    direction = -1; // to unpickle!
  }


  if (direction==1) { // just measure pickling speed
    cout << "*****Pickling: " << proto << endl;
    Array<char> buff;
    for (int ii=0; ii<times; ii++) {
      buff.clear();
      DumpValToArray(v, buff, ser); 
    }
  } else if (direction==-1) { // measure unpickling speed (with 1 pickle)
    cout << "*****UnPickling: " << proto << endl;
    Array<char> buff;
    DumpValToArray(v, buff, ser); 

    for (int ii=0; ii<times; ii++) {
      Val result;
      LoadValFromArray(buff, result, ser);
    }
  }

}
