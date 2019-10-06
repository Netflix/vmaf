#ifndef OCCONVERT_H_
#define OCCONVERT_H_

// These routines provide very fast implementation of converting Tabs
// to Arrs and vice-versa.

// There are also routines for converting OTabs and Tup to Tabs and
// Arr: these are provided for Backwards compatibility because PTOOLS
// 1.2.0 introduced OTab and Tup (which are unsupported in earlier
// versions): If you have have code that ONLY expects Arr and Tab,
// then this will handle the conversion for you.

// ///// Include
#include "ocval.h"


OC_BEGIN_NAMESPACE

// Forwards:  the main routines of this module

// Convert INPLACE the given Arr to a Tab: This assumes the given Val
// contains an Arr.  This should be a reasonably quick implementation,
// as no full copies are ever copied.  Do this operation recursively
// by default.  If keys_as_strings is set to "false" (default), when
// an Arr is converted to a Tab, all keys are the ints 0..n-1:
// otherwise keys are "0","1",.."n-1".
inline void ConvertArrToTab (Val& arr_must_be_in_here, 
			     bool recurse, bool keys_as_strings);

// Recursively convert any Arrs in v to Tabs.  The given Val does NOT
// have to be an Arr.
inline void ConvertAllArrsToTabs (Val& v, bool keys_as_strings);

// Convert INPLACE the given Tab to an Arr, if it makes sense (i.e.,
// the keys of the Tab look like contiguous integers).  This assumes
// the given Val contains a Tab.  Returns true if the operation
// succeeds, false otherwise (and has no effect on the Tab).  By
// default, do this operation recursively 
inline bool ConvertTabToArr (Val& tab_must_be_in_here, bool recurse);

// Recursively convert any Tabs in v to Arrs.  The given Val does NOT
// have to be an Tab.
inline void ConvertAllTabsToArrs (Val& v);


// In PicklingTools 1.2.0, we introduced OTab and Tup and int_n and
// int_un, but previous versions don't support those.  In
// compatibility mode for most serializations, the OTab is converted
// to a Tab, the Tup to an Arr, the int_n/int_un to a string.  This
// will do that conversion for you explicitly INPLACE if you need to.
// This will recursively convert ALL OTabs to Tabs Tups to Arrs.  Note
// that this preserves allocators, Proxys, and all general structure.
inline void ConvertAllOTabTupBigIntToTabArrStr (Val& v);


// Turn into a list:  For POD arrays, just copies each of the elements
// out.  
inline void AsList (Val& v, Val& result);
inline void AsList (const Val& v, Val& result, bool);

// _ routines are usually helper routines to implement.

// Conventional way: makes a shallow copy of the original top-level
// Tab top OTab.
inline void ConvertTabToOTab (const Tab& orig, OTab& result)
{
  for (It ii(orig); ii(); ) {
    result[ii.key()] = ii.value();
  }
}

// Conventional way: makes a shallow copy of the original top-level
// OTab to Tab.
inline void ConvertOTabToTab (const OTab& orig, Tab& result)
{
  for (It ii(orig); ii(); ) {
    result[ii.key()] = ii.value();
  }
}


#define NON_DESTRUCTIVE false
#define DESTRUCTIVE true

struct ConvertContext_ {
  ConvertContext_ () : destructive(DESTRUCTIVE) { }
  // Lookup proxy handles by void* to see if it has been already
  // processed: the handle is of the original: when a OTab or Tup is
  // converted, a brand new deep-copy is made of the proxy, because we
  // can't change the original proxy in case other people were
  // referring to it.  So, the proxy looked up is the NEW copy with
  // the NEW Tab or Arr. (for the resp. OTab and Tup).  Kind of
  // inefficient, but at least it works in all circumstances.
  AVLHashT<void*, Proxy, 8> already_processed;

  // If you are destructive, you are destroying the original thing to
  // convert, otherwise you are making a new copy.
  bool destructive;
};

// Forwards 
inline void ConvertAllOTabTupBigIntToTabArrStr_ (Val& v, ConvertContext_& cc);


template <class TTT>
inline void ConvertAllOTabTupBigIntToTabArrStr_ (TTT& o, 
						 Tab& t, 
						 ConvertContext_& cc)
{
  if (cc.destructive) {
    for (It ii(o); ii(); ) {
      Val& key = const_cast<Val&>(ii.key());
      Val& value = ii.value();
      ConvertAllOTabTupBigIntToTabArrStr_(value, cc);
      t.swapInto(key, value);
    }
  } else {
    bool old = cc.destructive;
    cc.destructive = DESTRUCTIVE;
    for (It ii(o); ii(); ) {
      const Val& key = ii.key();
      Val& value = ii.value();

      Val& value_ref = t[key] = Val();
      Val value_copy = value;
      ConvertAllOTabTupBigIntToTabArrStr_(value_copy, cc);
      value_copy.swap(value_ref);

    }
    cc.destructive = old;
  }
}

template <class TTT>
inline  void ConvertAllOTabTupBigIntToTabArrStr_ (TTT& u, Arr& a, 
						     ConvertContext_& cc)
{
  if (cc.destructive) {
    const int len = u.length();
    for (int ii=0; ii<len; ii++) {
      Val& tup_entry = u(ii);
      ConvertAllOTabTupBigIntToTabArrStr_(tup_entry, cc);
      a.append(Val());
      tup_entry.swap(a(ii));
    }
  } else {
    bool old = cc.destructive;
    cc.destructive = DESTRUCTIVE;
    const int len = u.length();
    for (int ii=0; ii<len; ii++) {
      a.append(u(ii));
      ConvertAllOTabTupBigIntToTabArrStr_(a(ii), cc);
    }
    cc.destructive = old;
  }
}


// Entry routine to convert recursively: we have to call other
// routines to do a true DFS
inline void ConvertAllOTabTupBigIntToTabArrStr (Val& v)
{
  // Keep proxy lookups by handle: after we have converted a Proxy, we
  // note it is done so we don't try to convert it again.
  ConvertContext_ cc;
  
  // Have to keep track of every single proxy we've seen and converted
  // so that we DO NOT convert them again.
  ConvertAllOTabTupBigIntToTabArrStr_(v, cc);
}


template <class FROM, class TO>
inline void ConvertOTabAndTupHelper_ (Val& v,
				      FROM* /*from*/, TO* /*to*/, 
				      ConvertContext_& cc)
{
  // Proxies have to be handled like a DFS, where we mark what we've
  // already seen to make sure we don't process it again.
  Proxy& p = v;
  void* handle = p.handle_;

  if (!cc.already_processed.contains(handle)) { 
    FROM& o = p;
    Proxy new_proxy = ProxyCopy(p, (TO*)0);
    cc.already_processed[handle] = new_proxy;
    TO& t = new_proxy;
    
    bool old = cc.destructive;
    cc.destructive = NON_DESTRUCTIVE;
    ConvertAllOTabTupBigIntToTabArrStr_(o, t, cc);
    cc.destructive = old;
  }
  // Install new proxy: may be brand new copy, or previously converted 
  v = cc.already_processed(handle);  
}

// Only really worries about OTab and Tup proxies which have to change
// inplace: by this, we actually create a new Proxy and "keep" the old
// one.
inline void ConvertOTabAndTupProxy_ (Val& v, ConvertContext_& cc)
{
  switch (v.tag) {
  case 't': ConvertOTabAndTupHelper_(v, (Tab*)0, (Tab*)0, cc); break;
  case 'o': ConvertOTabAndTupHelper_(v, (OTab*)0,(Tab*)0, cc); break;
  case 'u': ConvertOTabAndTupHelper_(v, (Tup*)0, (Arr*)0, cc); break;
  case 'n': 
    if (v.subtype == 'Z') {
      ConvertOTabAndTupHelper_(v, (Arr*)0, (Arr*)0, cc);
    } else {
      ; // No conversion for POD Arrays
    }
    break;
  default:
    throw logic_error("Proxy conversion problems");
  }
}


inline void ConvertAllOTabTupBigIntToTabArrStr_ (Val& v, 
						    ConvertContext_& cc)
{
  // Handle proxies special
  if (IsProxy(v)) {
    ConvertOTabAndTupProxy_(v, cc);
  }

  // If you see an OTab, turn it into a Tab
  else if (v.tag=='o') {
    OTab& o = v;
    Val newval = Tab(o.allocator()); // TODO: Allocators
    Tab& t = newval;
    ConvertAllOTabTupBigIntToTabArrStr_(o, t, cc);
    newval.swap(v);      
  }
  
  // If you see a Tup, turn it into an Arr
  else if (v.tag=='u') {
    Tup& u = v;
    Val newval = Arr(u.allocator());
    Arr& a = newval; 
    ConvertAllOTabTupBigIntToTabArrStr_(u, a, cc);
    newval.swap(v);
  }
  // int_n
  else if (v.tag=='q') {
    int_n* ip=(int_n*)&v.u.q;
    string s = ip->stringize();
    v = s;
  }
  // in_un
  else if (v.tag=='Q') {
    int_un* ip=(int_un*)&v.u.Q;
    string s = ip->stringize();
    v = s;
  }

  // Otherwise, you have to look recursively inside other things
  // JUST to make sure they don't have any OTab or Tup!
  else if (v.tag=='t' || (v.tag=='n' && v.subtype=='Z')) {
    for (It ii(v); ii(); ) {
      ConvertAllOTabTupBigIntToTabArrStr_(ii.value(), cc);
    }
  }

  // Otherwise, no need to do anything
}


// Helper routine: assumes we are doing recursive conversion of all
// Arrs to Tabs, if possible
inline void ConvertAllArrsToTabs (Val& v, bool keys_as_strings=false)
{
  // Iterate through as a Tab or OTab or Tup or Arr
  if (v.tag=='t' || v.tag=='o' || v.tag=='u' || (v.tag=='n'&&v.subtype=='Z')) {
    for (It ii(v); ii(); ) {
      Val&       value = ii.value();
      ConvertAllArrsToTabs(value, keys_as_strings);
    }
    if (v.tag=='n'&&v.subtype=='Z') {
      ConvertArrToTab(v, true, keys_as_strings);
    }
  }
  // Base case: no iteration
}


// main routine for conversion
inline void ConvertArrToTab (Val& arr_here, bool recurse=true,
			     bool keys_as_strings=false)
{
  Arr& a = arr_here;
  Val tab_here = Tab();
  if (IsProxy(arr_here)) tab_here.Proxyize(); // Maintain proxyiness in convert
  Tab& t = tab_here;

  int alen = a.length();
  for (int ii=0; ii<alen; ii++) {
    Val& v_in_arr = a[ii];
    if (recurse) {
      ConvertAllArrsToTabs(v_in_arr, keys_as_strings);
    } 

    // Assertion: Have a Val want to to put in the Tab
    if (keys_as_strings) {
      Val& v_in_tab = t[Stringize(ii)] = None;
      v_in_tab.swap(v_in_arr);
    } else {
      Val& v_in_tab = t[ii] = None;
      v_in_tab.swap(v_in_arr);
    }
  }
  tab_here.swap(arr_here); // Convert, in place!
}


// Helper routine: assumes we are doing recursive conversion of all
// Tabs to Arrs, if possible
inline void ConvertAllTabsToArrs (Val& v)
{
  // Iterate through as a Tab or OTab or Tup or Arr
  if (v.tag=='t' || v.tag=='o' || v.tag=='u' || (v.tag=='n'&&v.subtype=='Z')) {
    for (It ii(v); ii(); ) {
      Val&       value = ii.value();
      ConvertAllTabsToArrs(value);
    }
    if (v.tag=='t') {
      ConvertTabToArr(v, true);
    }
  }
  // Base case: no iteration
}

// Helper routine which detects if a key is an in integer or can VERY
// EASILY be converted to an integer. ("0", "1", "123").  If it can
// convert it, it will do the conversion and set value correctly.
// Otherwise, this routines routine returns false and value is NOT
// set.
inline bool CanBeInt (const Val& v, int& value)
{
  // Any of these Val types are probably not a good match for a
  // 'simple' int.  You could argue float and double (f&d), but we
  // won't.
  static char bad_tags[]="tnZfdFDbou";
  const char tag = v.tag;
  for (char* bp=bad_tags; *bp; bp++) {
    if (tag==*bp) return false;
  }

  // Assertion: string or int type.
  if (tag=='a') { // string
    string s = v;
    const int s_len = s.length();
    for (int ii=0; ii<s_len; ii++) {
      if (!isdigit(s[ii])) {
	return false;
      }
    }
    // Assertion: all 0..9 in string, fall through and convert
  }
  // Assertion: valid type to convert to int
  value = v;
  return true;
}


// main routine for conversion
inline bool ConvertTabToArr (Val& tab_here, bool recurse=true)
{
  Tab& t = tab_here;

  // Record the pointer: we don't want to copy until we are SURE they
  // entire Tab can be converted, but we also don't want to do key
  // conversion to ints again.
  Array<Val*> there(t.entries());
  there.fill(0);
  
  // Iterate through and see if each entry is a "number" that will be
  // contiguous in a final array.
  for (It ii(t); ii(); ) {
    const Val& key = ii.key();
    Val& value = ii.value();

    // Recursively try the ones underneath, if specified
    if (recurse) {
      ConvertAllTabsToArrs(value);
    }

    // See if this key is "an int" and record where it was, as well as
    // where it's value is (so we don't have to reiterate or reconvert
    // the int to a string).
    int jj=-1;  // Can we convert to an int?
    if (CanBeInt(key, jj) && jj<int(there.entries()) && jj>=0 && there[jj]==0){
      there[jj] = &value;
    } else {
      return false;
    }
  }

  // If we make it here, all point to where they should be: So, we can
  // convert!  
  Val arr_here = Arr(t.entries());
  if (IsProxy(tab_here)) arr_here.Proxyize(); // maintain proxiness in convert
  Arr& a = arr_here;
  a.fill(None);
  const int a_len = a.length();
  for (int ii=0; ii<a_len; ii++) {
    there[ii]->swap(a[ii]);
  }

  // Finally swap the new and the old
  tab_here.swap(arr_here);

  return true;
}


#define OCASLISTCONVERT(T) { Array<T>& ad=v;T*a=ad.data(); for(int ii=0;ii<len;ii++){ardata[ii]=a[ii];} break; }
inline void AsList (Val& v, Val& result)
{
  // Quick case: Tuple or Arr 
  if ((v.tag=='n' && v.subtype=='Z') || (v.tag=='u')) {
    v.swap(result);
    result.tag = 'n'; result.subtype = 'Z';
    return;
  }

  // Containers:
  if (strchr("ont", v.tag)!=NULL) {

    // Set-up
    const int len = v.length();
    result = Arr(v.length());
    Arr& aresult = result;
    aresult.fill(None);
    Val* ardata = aresult.data();

    // n tag: almost there already!
    if (v.tag=='n') {
      switch (v.subtype) {
      case 's': OCASLISTCONVERT(int_1); 
      case 'S': OCASLISTCONVERT(int_u1); 
      case 'i': OCASLISTCONVERT(int_2); 
      case 'I': OCASLISTCONVERT(int_u2); 
      case 'l': OCASLISTCONVERT(int_4); 
      case 'L': OCASLISTCONVERT(int_u4); 
      case 'x': OCASLISTCONVERT(int_8); 
      case 'X': OCASLISTCONVERT(int_u8); 
      case 'f': OCASLISTCONVERT(real_4); 
      case 'd': OCASLISTCONVERT(real_8);
      case 'c': OCASLISTCONVERT(cx_t<int_1>); break;
      case 'C': OCASLISTCONVERT(cx_t<int_u1>); break;
      case 'e': OCASLISTCONVERT(cx_t<int_2>); break;
      case 'E': OCASLISTCONVERT(cx_t<int_u2>); break;
      case 'g': OCASLISTCONVERT(cx_t<int_4>); break;
      case 'G': OCASLISTCONVERT(cx_t<int_u4>); break;
      case 'h': OCASLISTCONVERT(cx_t<int_8>); break;
      case 'H': OCASLISTCONVERT(cx_t<int_u8>); break; 
      case 'F': OCASLISTCONVERT(complex_8); 
      case 'D': OCASLISTCONVERT(complex_16); 
      default : throw runtime_error("Only arrays of POD");
      }
    }
    // Other containers
    else if (v.tag=='o' || v.tag=='t') {
      int jj = 0;
      for (It ii(v); ii(); jj++) {
	const Val& key = ii.key();
	Val& value = ii.value();
	// Swap the values in
	Val item = Tup(key, None);
	item[1].swap(value);
	item.swap(ardata[jj]);
      }
    }
  } 
  // Not a container
  else {
    result = Arr();
    result.append(None);
    result[0].swap(v);
  }
}

inline void AsList (const Val& v, Val& result, bool)
{
  // Plain arrays don't have any performance advantage
  // to swapping, they have to copy anyway, so no reason to
  // make yet ANOTHER copy
  if (v.tag=='n' && v.subtype != 'Z') {
    Val& nonconst_v = const_cast<Val&>(v);
    AsList(nonconst_v, result);
  }
  // Just copy
  else {
    Val vcopy(v);
    AsList(vcopy, result);
  }
}

OC_END_NAMESPACE

#endif  // OCCONVERT_H_
