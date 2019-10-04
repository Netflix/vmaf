
#include "chooseser.h"

// Compare two tables based on serialization
void Compare (const Val& in, const Val& out, Serialization_e ser, 
	      const Val& debug_info)
{
  // Serialize_NONE will only give back same thing if it was a string
  if (ser==SERIALIZE_NONE) {
    Str s = in;
    if (s != out) {
      cout << s << endl;
      in.prettyPrint(cout);
      out.prettyPrint(cout);
      debug_info.prettyPrint(cout);
      cerr << "Serialization Problem:" << int(ser) << endl;
      exit(1);
    }
  } else {
    
    if (in != out) {
      in.prettyPrint(cout);
      out.prettyPrint(cout);
      debug_info.prettyPrint(cout);
      cerr << "Serialization Problem:" << int(ser) << endl;
      exit(1);
    }
  }
} 

// Just test that the basics work for given serialization
void Trial (Serialization_e ser)
{
  cout << "*** Testing:" << int(ser) << endl;
  Val in = Tab("{'a':1, 'b':2.2, 'c':None, 'd': [1,2,'three'] }");
  
  Array<char> buff;
  DumpValToArray(in, buff, ser);
  Val out;
  LoadValFromArray(buff, out, ser);

  Compare(in, out, ser, 
	  in);
}

// These trials test how OTab, Tup and BigInts dump
void TrialConv (Serialization_e dump_ser, const Val& in, int load_ser_int = -1)
{
  Serialization_e load_ser = 
    load_ser_int == -1 ? dump_ser : Serialization_e(load_ser_int);
  cout << "*** Testing: dump_ser" << int(dump_ser) << " load_ser:" << load_ser << endl;
  
  // Try all combinations of converting in and converting out
  for (int ii=0; ii<2; ii++) {
    bool convert_on_load = bool(ii);
    for (int jj=0; jj<2; jj++) {
      // Load and dump, and see what converts 
      bool convert_on_dump = bool(jj);
      cout << "convert_on_dump:" << convert_on_dump 
	   << " convert_on_load:" << convert_on_load << endl;;

      Array<char> buff;
      DumpValToArray(in, buff, dump_ser, AS_LIST, convert_on_dump);
      Val out;
      LoadValFromArray(buff, out, load_ser, AS_LIST, convert_on_load);
      
      // neither coonvert: as is
      if (!convert_on_load && !convert_on_dump) {
	if (dump_ser==SERIALIZE_M2K) {
	  cout << "... M2k has no equivalent OTab, Tup, or BigInt structure ... continuing." << endl;
	  continue;
	}
	Compare(in, out, dump_ser, in);
      }
      // both convert
      else if (convert_on_load && convert_on_dump) {

	Val new_in = in;
	ConvertAllOTabTupBigIntToTabArrStr(new_in);
	Compare(new_in, out, dump_ser, in);
      }
      // Only converting on load
      else if (convert_on_load && !convert_on_dump) {
	Val new_in = in;
	ConvertAllOTabTupBigIntToTabArrStr(new_in);
	Compare(new_in, out, dump_ser, in);
      }
      // Only converting when dumping
      else  if (convert_on_dump && !convert_on_load) {
	Val new_in = in;
	ConvertAllOTabTupBigIntToTabArrStr(new_in);
	Compare(new_in, out, dump_ser, in);
      }

    }
  }
}


void TrialFile (Serialization_e ser)
{
  cout << "*** Testing File:" << int(ser) << endl;
  Val in = Tab("{'a':1, 'b':2.2, 'c':None, 'd': [1,2,'three'] }");
  string filename = "/tmp/junk"+Stringize(int(ser));
  const char* fn = filename.c_str();

  DumpValToFile(in, fn, ser);
  Val out;
  LoadValFromFile(fn, out, ser);

  Compare(in, out, ser, in);
}

int main ()
{

  Trial(SERIALIZE_P0);
  Trial(SERIALIZE_P0_OLDIMPL);
  Trial(SERIALIZE_P2);
  Trial(SERIALIZE_P2_OLDIMPL);
  Trial(SERIALIZE_P2_OLD);
  Trial(SERIALIZE_M2K);
  Trial(SERIALIZE_OC);
  Trial(SERIALIZE_TEXT);
  Trial(SERIALIZE_PRETTY);
  Trial(SERIALIZE_NONE);

  Val in = OTab("o{}");
  Val copy = in;
  ConvertAllOTabTupBigIntToTabArrStr(copy);
  cout << in << endl << copy << endl;
  TrialConv(SERIALIZE_P0, in);
  TrialConv(SERIALIZE_P0_OLDIMPL, in);
  TrialConv(SERIALIZE_P2, in);
  TrialConv(SERIALIZE_P2_OLDIMPL, in);
  TrialConv(SERIALIZE_P2_OLD, in);
  TrialConv(SERIALIZE_M2K, in);
  TrialConv(SERIALIZE_OC, in);
  TrialConv(SERIALIZE_TEXT, in);
  TrialConv(SERIALIZE_PRETTY, in);
  // TrialConv(SERIALIZE_NONE, in); // No conversion happens during NONE

  in = OTab("o{'a':1}");
  copy = in;
  ConvertAllOTabTupBigIntToTabArrStr(copy);
  cout << in << endl << copy << endl;
  TrialConv(SERIALIZE_P0, in);
  TrialConv(SERIALIZE_P0_OLDIMPL, in);
  TrialConv(SERIALIZE_P2, in);
  TrialConv(SERIALIZE_P2_OLDIMPL, in);
  TrialConv(SERIALIZE_P2_OLD, in);
  TrialConv(SERIALIZE_M2K, in);
  TrialConv(SERIALIZE_OC, in);
  TrialConv(SERIALIZE_TEXT, in);
  TrialConv(SERIALIZE_PRETTY, in);
  // TrialConv(SERIALIZE_NONE, in); // No conversion happens during NONE


  in = Tup(1,2);
  TrialConv(SERIALIZE_P0, in);
  TrialConv(SERIALIZE_P0_OLDIMPL, in, SERIALIZE_P0); // P0_OLD_IMPL can't load Tuples well
  TrialConv(SERIALIZE_P2, in);
  TrialConv(SERIALIZE_P2_OLDIMPL, in, SERIALIZE_P2); // P2_OLD_IMPL can't load Tuples well
  //TrialConv(SERIALIZE_P2_OLD, in);  // TODO: do we support anymore?
  TrialConv(SERIALIZE_M2K, in);
  TrialConv(SERIALIZE_OC, in);
  TrialConv(SERIALIZE_TEXT, in);
  TrialConv(SERIALIZE_PRETTY, in);
  // TrialConv(SERIALIZE_NONE, in); // No conversion happens during NONE


  in = StringToBigInt("123456789123456789123456789");
  copy = in;
  ConvertAllOTabTupBigIntToTabArrStr(copy);
  cout << in << endl << copy << endl;
  TrialConv(SERIALIZE_P0, in);
  TrialConv(SERIALIZE_P0_OLDIMPL, in, SERIALIZE_P0); // P0_OLD_IMPL can't load it_n well
  TrialConv(SERIALIZE_P2, in);
  TrialConv(SERIALIZE_P2_OLDIMPL, in, SERIALIZE_P2); // P2_OLD_IMPL can't load int_un well
  // TrialConv(SERIALIZE_P2_OLD, in); // TODO: do we support anymore?
  TrialConv(SERIALIZE_M2K, in);
  TrialConv(SERIALIZE_OC, in);
  TrialConv(SERIALIZE_TEXT, in);
  TrialConv(SERIALIZE_PRETTY, in);
  // TrialConv(SERIALIZE_NONE, in); // No conversion happens during NONE

  in = OTab("o{'a':1, 'b':(1), 'c':(2,2.2), 'd':(3,3,3), 'e':(4,4,4,4), 'f':(5,5.5,5,(5+5j),50000000000000000000), 'c':o{}, 'd': (o{},o{'a':(1)}) }");
  TrialConv(SERIALIZE_P0, in);
  TrialConv(SERIALIZE_P0_OLDIMPL, in, SERIALIZE_P0); // P0_OLD_IMPL can't load it_n well
  TrialConv(SERIALIZE_P2, in);
  TrialConv(SERIALIZE_P2_OLDIMPL, in, SERIALIZE_P2); // P2_OLD_IMPL can't load it_n well
  // TrialConv(SERIALIZE_P2_OLD, in); // TODO: Do we support?
  TrialConv(SERIALIZE_M2K, in);
  TrialConv(SERIALIZE_OC, in);
  TrialConv(SERIALIZE_TEXT, in);
  TrialConv(SERIALIZE_PRETTY, in);
  //TrialConv(SERIALIZE_NONE, in);

  TrialFile(SERIALIZE_P0);
  TrialFile(SERIALIZE_P0_OLDIMPL);
  TrialFile(SERIALIZE_P2);
  TrialFile(SERIALIZE_P2_OLDIMPL);
  TrialFile(SERIALIZE_P2_OLD);
  TrialFile(SERIALIZE_M2K);
  TrialFile(SERIALIZE_OC);
  TrialFile(SERIALIZE_TEXT);
  TrialFile(SERIALIZE_PRETTY);
  TrialFile(SERIALIZE_NONE);
}
