
#include "ocport.h"

#define MIDAS_COMPILER_TEMPLATES
#include "valpython.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif



using namespace std;

#include <stdio.h>

int main (int argc, char* argv[])
{
  if (argc != 4) {
    cerr << "Usage: filename-to-write-to 1(uses Numeric)|0(no Numeric)|2(NumPy) 1(compat serializtion)|0(no compat)" << endl;
    cerr << " ... compatibility mode: 1 means that Tuples and OTabs serialize like Arrs and Tabs (resp.), 0 means that Tuples and OTabs serialize as Python Tuples and Python OrderedDicts (resp.)." << endl;
    return 0;
  }

  string name = argv[1];
  cout << "Writing " << name << endl;
  ArrayDisposition_e arr_disp;
  char c = argv[2][0];
  cerr << "c=" << c << endl;
  arr_disp = c=='0' ? AS_LIST : AS_NUMERIC;
  arr_disp = c=='2' ? AS_NUMPY: AS_LIST;
  cerr << "arr_disp" << arr_disp << endl;
  //bool uses_numeric = (argv[2][0]=='1');
  bool compat = (argv[3][0]=='1');

  Val vv = Tab();
  Tab& t = vv;
  t["the"] = 1;
  t["brown"] = 123.456;
  Tab sub;
  sub["sub"] = "table";
  sub[6] = 66;
  Arr l;
  l.append(1);
  l.append(2);
  l.append("three");
  sub["mylist"] = l;
  t["keyme"] = sub;
  t["pi"]    = 3.1415926535897932846;
  t["phase"]       = complex_16(6.7, 8.9);
  t["phase_small"] = complex_8(10.10, 12.12);
  // test out the arrays
  t["arr"] = Array<int_4>(10);
  Array<int_4>& aa = t["arr"];
  for (int ii=0; ii<2; ii++) {
    aa.append(ii);
  }
  t["carr"] = Array<complex_16>(10);
  Val vc = complex_16(7.8, 9.5);
  Val va = Array<complex_16>(10);
  Array<complex_16>& aaa = va;
  for (int kk=0; kk<3; kk++) {
    aaa.append(complex_16(kk, kk+10));
  }
  t["more"] = Tab();
  t["more"]["cx"] = vc;
  t["more"]["cx_a"] = va;
  t["more"]["cx_float"] = complex_8(123456789, 987654321.0);
  t["more"]["cx_float_array"] = Array<complex_8>(10);
  Array<complex_8>& acx = t["more"]["cx_float_array"];
  for (int ll=0; ll<3; ll++) {
    acx.append(complex_8(100+ll, 100*ll));
  }
  t["same_string"] = "same_string";
  t[0] = "same_string";
  t[1] = "same_string";
  t["Noned"] = None;
  t["Truthman"] = true;
  t["Not Truth man"] = false;
  t["tuple"] = Tup(1,"hello",Arr(),Tup(1,2,3),Tab("{'a':1, 'b':('10',12)}"));
  t["tuple"][0] = OTab("o{'a':1, 'b':2, 'c':3}");
  t["empty otab"] = OTab();
  t["empty_tuple"] = Tup();
  t["proxy me"] = new Tup(1,2,3);
  t["two of them"] = Tup(t["proxy me"], t["proxy me"]);
  cout << t << endl;
  {
    PythonPickler<Val> pp(name, arr_disp);
    pp.compatibility(compat);
    pp.dump(vv);
  }

  return 1;
}
