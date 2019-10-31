
#include "ocval.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

int main ()
{
  { cout << "Empty ArrIt" << endl;
    ArrIt<Val> a;
    while (a()) {
      cout << a.key() << " " << a.value() << endl;
    }
  }

  {
    Arr a("[1, 2.2, 'three']");
    cout << a << endl;
    ArrIt<Val> ii(a);
    while (ii()) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  {
    Arr a("['single']");
    cout << "Single ArrIt:" << a << endl;
    ArrIt<Val> ii(a);
    while (ii()) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  {
    Arr a;
    cout << "Empty ArrIt:" << a << endl;
    ArrIt<Val> ii(a);
    while (ii()) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  { cout << "Empty ArrSIt" << endl;
    ArrSit<Val> a;
    while (a()) {
      cout << a.key() << " " << a.value() << endl;
    }
  }

  {
    Arr a("[1, 2.2, 'three']");
    cout << "Some ArrSit:" << a << endl;
    ArrSit<Val> ii(a);
    while (ii()) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  {
    Arr a("['single']");
    cout << "Single ArrSit:" << a << endl;
    ArrSit<Val> ii(a);
    while (ii()) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  {
    Arr a;
    cout << "Empty ArrSit:" << a << endl;
    ArrSit<Val> ii(a);
    while (ii()) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  { cout << "Empty TabIt" << endl;
    TabIt a;
    while (a()) {
      cout << a.key() << " " << a.value() << endl;
    }
  }

  { 
    Tab t;
    cout << "Empty tab" << t << endl;
    TabIt a(t);
    while (a()) {
      cout << a.key() << " " << a.value() << endl;
    }
  }

  { 
    Tab t("{'a':1, 'b':2.2, 'c':'three'}");
    cout << t << endl;
    TabIt ii(t);
    while (ii()) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  { 
    Tab t("{'a':1}");
    cout << "Single" << t << endl;
    TabIt ii(t);
    while (ii()) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  { cout << "Empty TabSit" << endl;
    TabSit a;
    while (a()) {
      cout << a.key() << " " << a.value() << endl;
    }
  }
  { 
    Tab t;
    cout << "Empty tab for TabSit" << t << endl;
    TabSit a(t);
    while (a()) {
      cout << a.key() << " " << a.value() << endl;
    }
  }

  { 
    Tab t("{'a':1, 'b':2.2, 'c':'three'}");
    cout << "tabSit:" << t << endl;
    TabSit ii(t);
    while (ii()) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  { 
    Tab t("{'a':1}");
    cout << "Single TabSit" << t << endl;
    TabSit ii(t);
    while (ii()) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  {
    cout << "It; Arr empty" << endl;
    Arr a;
    for (It ii(a); ii(); ) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  {
    Arr a("['single']");
    cout << "It; single" << endl;
    for (It ii(a); ii(); ) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  { 
    Arr a("[1,'three', 2.2]");
    cout << "It; single" << endl;
    for (It ii(a); ii(); ) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  {
    cout << "SIt; Arr empty" << endl;
    Arr a;
    for (Sit ii(a); ii(); ) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  {
    Arr a("['single']");
    cout << "Sit; single" << endl;
    for (Sit ii(a); ii(); ) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  { 
    Arr a("[1,'three',2.2]");
    cout << "Sit; single" << endl;
    for (Sit ii(a); ii(); ) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }


  {
    cout << "It; Tab empty" << endl;
    Tab a;
    for (It ii(a); ii(); ) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  {
    Tab a("{'single':1}");
    cout << "It; single" << endl;
    for (It ii(a); ii(); ) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  { 
    Tab a("{1:1,'three':'three', 2.2:2.2}");
    cout << "It; single" << endl;
    for (It ii(a); ii(); ) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  {
    cout << "SIt; Tab empty" << endl;
    Tab a;
    for (Sit ii(a); ii(); ) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  {
    Tab a("{'single':1}");
    cout << "Sit; single" << endl;
    for (Sit ii(a); ii(); ) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  { 
    Tab a("{1:1,'three':3,2.2:'two'}");
    cout << "Sit; single" << endl;
    for (Sit ii(a); ii(); ) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  {
    try {
      Val v;
      for (It ii(v); ii(); ) {
	cout << ii.key() << " " << ii.value() << endl;
      }
    } catch (exception& e) {
      cout << "Expected exception:" << e.what() << endl;
    }
  }


  {
    try {
      Val v = Array<int>(10);
      for (It ii(v); ii(); ) {
	cout << ii.key() << " " << ii.value() << endl;
      }
    } catch (exception& e) {
      cout << "Expected exception:" << e.what() << endl;
    }
  }


  {
    Proxy p = new Tab("{'a':1, 'b':2 }");
    for (It ii(p); ii(); ) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }


  {
    Proxy p = new Tab("{'a':1, 'b':2 }");
    for (Sit ii(p); ii(); ) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  {
    Proxy p = new Arr("['a',1,'b',2 ]");
    //Arr& a = p; 
    for (It ii(p); ii(); ) {
      cout << ii.key() << " ";
      try {
	Val& v = ii.value();
	if (v.tag=='*') exit(1); // Dumb test
	cout << ii.value() << endl;
      } catch (exception& e) {
	cerr << e.what() << endl;
      }
    }
  }

  {
    Proxy p = new Arr("['a',1,'b',2 ]");
    for (Sit ii(p); ii(); ) {
      cout << ii.key() << " " << ii.value() << endl;
    }
  }

  {
    Proxy p = new OTab("o{'a':1,'b':2}");
    for (It ii(p); ii(); ) {
      cout << ii.key() << " " << ii.value() << endl;
    }
    cout << p  << endl;
  }


  {
    Proxy p = new OTab("o{'b':1,'a':2}");
    for (Sit ii(p); ii(); ) {
      cout << ii.key() << " " << ii.value() << endl;
    }
    cout << p  << endl;
  }

  {
    Proxy p = new Tup('a',1,'b',2);
    for (It ii(p); ii(); ) { 
      cout << ii.key() << " " << ii.value() << endl;
    }
    cout << p  << endl;
  }


  {
    Proxy p = new Tup('a',1,'b',2);
    for (Sit ii(p); ii(); ) {
      cout << ii.key() << " " << ii.value() << endl;
    }
    cout << p  << endl;
  }

}
