#ifndef OCCONFORMS_H_

// Test to see if an instance of a Val conforms to the specification
// prototype.  For example:
//
// // EXAMPLE 1
// ///// At a simple level, we want to exactly match a prototype:
// Tab instance  = "{'a':1, 'b':2.2, 'c':'three'}"; // value to check
// Tab prototype = "{'a':0, 'b':0.0, 'c':""}";  // prototype to check against
// if (Conforms(instance, prototype, true, EXACT_MATCH)) {
//     ... should be true: all keys match, and value TYPES match
// }
// (1) Note that the instance has all the same keys as the prototype, so it
//     matches 
// (2) Note that on the prototype table, that the VALUES aren't important,
//     it's only matching the the TYPE of the val

// // EXAMPLE 2
// //// We may not want to necesssarily need all keys in the prototype
// Tab instance1 = "{'a':1, 'b':2.2, 'c':'three', 'd':777 }";
// Tab prototype1= "{'a':0, b:0.0 }";
// if (Conforms(instance1, prototype1, false, EXACT_MATCH)) {
//      ... should be true: instance has all keys of prototype 
// }
// (1) Note that the instance has more keys than the prototype, but that's
//     okay because we specified exact_structure to be false.
// (2) by setting EXACT_MATCH, all the types of the values that are
//     compared MUST match (not the value just the types of the values)

// // EXAMPLE 3
// //// If you just want the structure, but don't care about the
// //// types of the keys, use None in the prototype.
// Tab instance2 = "{'a':1, 'b':2.2, 'c':'three'}";
// Tab prototype2= "{'a':None, 'b':None, 'c':None }";
// if (Conforms(instance2, prototype2, true, EXACT_MATCH)) {
//     ... should be true, only comparing keys
// }

// // EXAMPLE 4
// //// If you want to match value types, but want to be a little
// //// looser: sometimes your int is an int_4, sometimes an int_u4, etc.
// Tab instance3 = "{'a':1, 'b':2.2, 'c':'three'}";
// instance3["b"] = real_4(2.2); // Not double, but real! Still want to match
// Tab prototype3 ="{'a':0, 'b':0.0, 'c':'three'}";
// if (Conforms(instance3, prototype3, true, LOOSE_MATCH)) {
//     ... should be true because real_8(2.2) is a LOOSE_MATCH of real_4(2.2)
// }

#include "ocval.h"

OC_BEGIN_NAMESPACE

// Check and see if the given "instance" of a Tab, Arr, etc. conforms
// to the given structure of a "prototype" table.
// To match, the instance must contain at least the same structure
// as the prototype.  If the "exact_structure" flag is set, the
// keys and number must match exactly.  If the "match_types" flag is set, then
// the type of the values of the keys are checked too.  
// See the Conform_e see what kind of matching is done.
// Finally, most of the time, you want to return true or false
// to indicate the instance matches the spec---when debugging,
// why an instance failed to match spec, you can have the
// routine throw an exception with a MUCH more descriptive message
// about why the conformance did not match.   Frequently, you won't
// need that (and it's also expensive as it stringizes the 
// tables of interest), so it's best you only use that feature
// to help debug.

// Specify how *values* (not keys) in a table or Array must match
enum Conform_e { 
  EXACT_MATCH,   // int_4 must be int_4, etc

  LOOSE_MATCH,        // any int matches any int, 
                      // any uint matches any uint
                      // any int matches any int
                      // any real matches any real,
                      // any Tab/OTab matches any Tab/OTab
                      // any Tup/Arr matches any Tup/Arr

  LOOSE_STRING_MATCH, // As LOOSE_MATCH, but if the instance is 
                      // is a string but the prototype is of type x,
                      // call it a match.
                      // Consider: 
                      // instance = { 'a': "123" } prototype = {'a':1 }
                      // LOOSE_STRING_MATCH would match because 
                      // "123" is a string that *could* convert to an int.
};             

// Main routine for user to call: See conform_test.cc for more examples.
inline bool Conforms (const Val& instance, const Val& prototype,
		      bool exact_structure = false,
		      Conform_e type_match = EXACT_MATCH,
		      bool throw_exception_with_message = false);



// Create an informative error message and throw an exception
// with that error message
void InformativeError (const string& mesg,
		       const Val& instance, const Val& prototype, 
		       bool exact_structure,
		       Conform_e type_match)
{
  // stringize conform
  string conform;
  if (type_match==EXACT_MATCH) 
    conform = "EXACT_MATCH";
  else if (type_match==LOOSE_MATCH) 
    conform= "LOOSE_MATCH";
  else if (type_match==LOOSE_STRING_MATCH)  
    conform = "LOOSE_STRING_MATCH";
  else 
    throw runtime_error("Illegal conform?"+Stringize(int(type_match)));
  
  // message to output
  string message = 
   "*********FAILURE TO MATCH instance against prototype:\n" 
   " instance="+Stringize(instance)+" with type:"+string(1,instance.tag)+"\n"
   " prototype="+Stringize(prototype)+" with type:"+string(1,prototype.tag)+"\n"
   " exact_structure="+Stringize(exact_structure)+"\n" 
   " type_match="+conform+"\n"+
   mesg;
  // cerr << message << endl;
  throw runtime_error(message);
}

// Check to see if both are primitive.  Primitive in this context
// means 
inline bool IsPrimitiveType (const Val& v)
{
  // Faster to see if NOT a composite type
  // const char* primitive_tags = "silxSILXfdFDbZa";
  static const char composite_types[] = { 't', 'o', 'n', 'u' };
  const char tag = v.tag;
  for (size_t ii=0; ii<sizeof(composite_types); ii++) {
    if (tag==composite_types[ii]) return false;
  }
  return true;
}

inline bool IsCompositeType (const Val& v)
{
  // const char* primitive_tags = "silxSILXfdFDbZa";
  static const char composite_types[] = { 't', 'o', 'n', 'u' };
  const char tag = v.tag;
  for (size_t ii=0; ii<sizeof(composite_types); ii++) {
    if (tag==composite_types[ii]) return true;
  }
  return false;
}

// See if two primitive types match: if v2 is None, then no need
// to check (None means 'don't match').
inline bool PrimitiveMatch_(const Val& v1, const Val& v2, 
			    bool exact_structure, Conform_e type_match,
			    bool throw_exception_with_message)
{
  // Doesn't matter: v2 (prototype) is None which means don't consider
  // the type 
  if (v2.tag == 'Z') {
    return true;
  }

  // Make sure both primitive types
  if (!(IsPrimitiveType(v1) && IsPrimitiveType(v2))) {
    if (throw_exception_with_message) {
      InformativeError("Neither Type is a primitive type", v1, v2, 
		       exact_structure, type_match);
    } else {
      return false;
    }
  }

  // For primitive types, no need for nested checking
  bool exact_match = (v1.tag == v2.tag);
  if (exact_match) return true;  // easy case

  // Only if they don't match exactlty, figure out what to do
  if (type_match == EXACT_MATCH) {
    if (exact_match==false && throw_exception_with_message) {
      InformativeError("Requested Exact Match of two primitive types that "
		       "didn't match", v1, v2, exact_structure, type_match);
    } else {
      return exact_match;
    }
  } else if (type_match == LOOSE_MATCH || type_match == LOOSE_STRING_MATCH) {

    // any complexes are close enough
    if (OC_IS_CX(v1) && OC_IS_CX(v2)) return true;
    // any reals are close enough
    if (OC_IS_REAL(v1) && OC_IS_REAL(v2)) return true;
    // any ints are close enough
    if (OC_IS_INT(v1) && OC_IS_INT(v2)) return true;
    if (OC_IS_UINT(v1) && OC_IS_UINT(v2)) return true;
    // any uints and ints are good enough match
    if (OC_IS_INT(v1) && OC_IS_UINT(v2)) return true;
    if (OC_IS_UINT(v1) && OC_IS_INT(v2)) return true;

    // Try convert other type
    if (type_match == LOOSE_STRING_MATCH) {
      // Sometimes, things are stringized and SHOULD be ints/floats
      // but they are still strings.  We let these through.
      if (v1.tag == 'a') {
	return true;
      }
    }
    // If we make it all the way here, then they didn't match anyway
    if (throw_exception_with_message) {
      InformativeError("Even with loose match, they didn't match", v1, v2,
		       exact_structure, type_match);
    } else {
      return false;
    }
  } else {
    throw runtime_error("Unknown match type?");
  }
  return false;
}


// Helper function for just comparing Tables		       
template <class TAB1, class TAB2>
inline bool MatchTableHelper_ (const TAB1& instance, const TAB2& prototype, 
			       bool exact_structure,
			       Conform_e type_match,
			       bool throw_exception_with_message)
{
  if (exact_structure) {
    // must have same number of elements, and each element must match
    if (instance.entries() != prototype.entries()) {
      if (throw_exception_with_message) {
	InformativeError("instance and prototype do NOT have the same number"
			 "of elements and we have requested an exact match",
			 instance, prototype, exact_structure, type_match);
      } else {
	return false;
      }
    }
  }
  // Assertion: just have to see if instance has required keys
  // specified in prototype
  It it(prototype);  // TODO: Do we want to enforce two OTabs having the same order??
  while (it()) {
    const Val& prototype_key = it.key();
    Val& prototype_value = it.value();

    // If key is there, yay
    if (instance.contains(prototype_key)) {
      Val& instance_value = instance(prototype_key);
      if (prototype_value == None) { // no need to compare types
	// Just presence of key is good enough to keep moving
	;
      } else { // have to compare types
	bool one_value_result =  Conforms(instance_value, prototype_value, 
					  exact_structure, type_match, 
					  throw_exception_with_message);
	// Whoops, this single key Conformance failed.
	if (!one_value_result) { 
	  return false;           // Conforms will give long error if needed
	}
      }
    } 

    // Key NOT there ... bad mojo! 
    else {
      if (throw_exception_with_message) {
	InformativeError("The prototype has the key:"+Stringize(prototype_key)+
			 " but the instance does not.",
			 instance, prototype, exact_structure, type_match);
      } else {
	return false;
      }
    }
  }
  return true;
}


// See if two tables conform: front-end for MatchTableHelper_
inline bool TableMatch_ (const Val& v1, const Val& v2,
			 bool exact_structure,
			 Conform_e type_match,
			 bool throw_exception_with_message)
{
  // No pattern match needed
  if (v2.tag=='Z') {
    return true;
  }

  // Only tables
  if (!(v1.tag == 't' || v1.tag=='o' || v2.tag=='t' || v2.tag=='o')) {
    if (throw_exception_with_message) {
      InformativeError("TableMatch_ can only compare tables:",
		       v1, v2, exact_structure, type_match);
    } else {
      return false;
    }
  }
  
  // Same kind of table
  if (v1.tag == 't' && v2.tag == 't') {
    Tab& t1 = v1; Tab& t2 = v2;
    return MatchTableHelper_(t1, t2, exact_structure, type_match, 
			     throw_exception_with_message);
  }
  if (v1.tag == 'o' && v2.tag == 'o') {
    OTab& t1 = v1; OTab& t2 = v2;
    return MatchTableHelper_(t1, t2, exact_structure, type_match, 
			     throw_exception_with_message);
  }
  
  if (type_match == EXACT_MATCH) {
    if (throw_exception_with_message) {
      InformativeError("Not the same type for EXACT_MATCH", v1, v2,
		       exact_structure, type_match);
    } else {
      return false;  // because not same type!
    }
  }
  
  // Otherwise, handle slightly incompatible types
  if (v1.tag == 't' && v2.tag == 'o') {
    Tab& t1 = v1; OTab& t2 = v2;
    return MatchTableHelper_(t1, t2, exact_structure, type_match, 
			     throw_exception_with_message);
  }
  else if (v1.tag == 'o' && v2.tag == 't') {
    OTab& t1 = v1; Tab& t2 = v2;
    return MatchTableHelper_(t1, t2, exact_structure, type_match, 
			     throw_exception_with_message);
  }
  if (throw_exception_with_message) {
    InformativeError("TableMatch:: Only comparing tables-how did we get here?",
		     v1, v2, exact_structure, type_match);
  }
  return false;
}


// helper for handling Arr
inline bool MatchArrayHelper_ (const Arr& instance, const Arr& prototype,
			       bool exact_structure,
			       Conform_e type_match,
			       bool throw_exception_with_message)
{
  // Exact structure implies same number of elements
  if (exact_structure && (instance.length() != prototype.length())) {
    if (throw_exception_with_message) {
      InformativeError("Arrays don't match: different lengths",
		       instance, prototype, exact_structure, type_match);
    } else {
      return false;
    }
  }
  
  // So, not exact structure.  But if the instance has fewer
  // elements than the prototype, they cannot match
  if (instance.length() < prototype.length()) {
    if (throw_exception_with_message) {
      InformativeError("instance Arr has fewer elements than prototype Arr", 
		       instance, prototype, exact_structure, type_match);
    } else {
      return false;
    }
  }

  // Make sure all elements of prototype are found in instance
  const int proto_len = int(prototype.length());
  for (int ii=0; ii<proto_len; ii++) {
    const Val& prototype_value = prototype[ii];
    
    if (prototype_value==None) {
      continue; // just checking existance, move on
    } else {
      const Val& instance_value  = instance[ii];
      bool result = Conforms(instance_value, prototype_value, exact_structure, 
			     type_match, throw_exception_with_message);
      if (!result) { // Check above handles better error message
	return false;
      }
    }
  }
  
  // made it here: all keys matched prototype well enough: call it good.
  return true;
}


// Statically call and check 
inline bool MatchArr_ (const Arr& a1, const Val& v2,
		       bool exact_structure,
		       Conform_e type_match,
		       bool throw_exception_with_message)
{
  if (v2.tag == 'u') {
    Tup& t2 = v2;
    return MatchArrayHelper_(a1, t2.impl(), exact_structure, type_match,
			     throw_exception_with_message);
  } else if (v2.tag!='n') {
    if (throw_exception_with_message) {
      InformativeError("MatchArr_: Arr can't compare to a v2", 
		       a1, v2, exact_structure, type_match);
    } else {
      return false;
    }
  } else if (v2.tag=='n' && v2.subtype=='Z') {
    Arr& a2 = v2;
    return MatchArrayHelper_(a1, a2, exact_structure, type_match,
			     throw_exception_with_message);
  } else { // v2.tag == 'n', and POD data
    // TODO: for now, Array<POD> don't match Arr or tuple
    if (throw_exception_with_message) {
      InformativeError("MatchArr_: Arr can't compare to Array<POD>",
		       a1, v2, exact_structure, type_match);
    } else {
      return false;
    }
  }
  return false;
}

// Match helper for Arr and something else
inline bool MatchArr_ (const Val& v1, const Arr& a2,
		       bool exact_structure,
		       Conform_e type_match,
		       bool throw_exception_with_message)
{
  if (v1.tag == 'u') {
    const Tup& t1 = v1;
    return MatchArrayHelper_(t1.impl(), a2, exact_structure, type_match,
			     throw_exception_with_message);
  } else if (v1.tag!='n') {
    if (throw_exception_with_message) {
      InformativeError("MatchArr_: v1 can't compare to an Arr ",
		       v1, a2, exact_structure, type_match);
    } else {
      return false;
    }
  } else if (v1.tag=='n' && v1.subtype=='Z') {
    const Arr& a1 = v1;
    return MatchArrayHelper_(a1, a2, exact_structure, type_match,
			     throw_exception_with_message);
  } else { // v1.tag == 'n', and POD data
    // TODO: for now, Array<POD> don't match Arr or tuple
    if (throw_exception_with_message) {
      InformativeError("MatchArr_: Arr can't compare to Array<POD>",
		       v1, a2, exact_structure, type_match);
    } else {
      return false;
    }
  }
  return false;
}


// See if two tables conform: front-end for MatchArrayHelper_
inline bool ArrayMatch_ (const Val& v1, const Val& v2,
			 bool exact_structure,
			 Conform_e type_match,
			 bool throw_exception_with_message)
{
  // No pattern match needed
  if (v2.tag=='Z') {
    return true;
  }

  // Double check only arrays 
  if (!(v1.tag=='u' || v1.tag=='n' || v2.tag=='u' || v2.tag=='n')) {
    if (throw_exception_with_message) {
      InformativeError("ArrayMatch_: Only can handle arrays and tuples",
		       v1, v2, exact_structure, type_match);
    } else {
      return false;
    }
  }
  
  // Two tuples?
  if (v1.tag=='u' && v2.tag=='u') {
    Tup& t1 = v1; Tup& t2 = v2;
    return MatchArrayHelper_(t1.impl(), t2.impl(), exact_structure, type_match, 
			     throw_exception_with_message);
  } 

  // Two arrays?
  else if (v1.tag=='n' && v2.tag=='n') {

    // Both Arr
    if (v1.subtype=='Z' && v2.subtype=='Z') {
	return MatchArrayHelper_(v1, v2, exact_structure, type_match, 
				  throw_exception_with_message);
    } 
    
    // One of the two is a POD Array, the other is an Arr
    else if (v1.subtype=='Z' || v2.subtype=='Z') {
      // left is a POD array
      if (throw_exception_with_message) {
	InformativeError("POD Array doesn't match ARR", 
			 v1, v2, exact_structure, type_match);
      } else {
	return false; // For now, POD Array does not match Arr
      }
    }
    
    // Otherwise, both are pod arrays, so they'll match only if the array
    // types match
    else {
      if (v1.subtype==v2.subtype) return true;
      Val special1=complex_16(0);
      Val special2=complex_16(0);  // Make sure fill in vals with valid 0s
      special1.tag = v1.subtype;
      special2.tag = v2.subtype;  // Bad code: I depend on the impl of Val
      return PrimitiveMatch_(special1, special2, exact_structure, type_match,
			     throw_exception_with_message);
    }
  }

  // Some mix of tuples and arrays
  if (type_match == EXACT_MATCH) {
    if (throw_exception_with_message) {
      InformativeError("Requested EXACT_MATCH but we mixed arrays and tuples",
		       v1, v2, exact_structure, type_match);
    } else {
      return false;
    }
  }
  
  // Assertion: mixed tuples and arrays
  if (v1.tag=='u') { // v2.tag == 'n'
    Tup& t1 = v1;
    const Arr& a1 = t1.impl();
    return MatchArr_(a1, v2, exact_structure, type_match, 
		     throw_exception_with_message);
  } else if (v2.tag == 'u') { // v1.tag=='n'
    Tup& t2 = v2;
    const Arr& a2 = t2.impl();
    return MatchArr_(v1, a2, exact_structure, type_match, 
		     throw_exception_with_message);
  } else {
    throw runtime_error("ArrayMatch_: only expected mix of array and tuples");
  }
  
}



// See if the instance conforms to the structure of the prototype. 
// Parameters:
//  instance is the Val to check for conformance
//  prototype is the Val that specifies the structure of a 'valid' entry
//  exact structure (if true) means the structure must match:
//     TRUE: Tab/OTab must have the same number of *keys* with same names
//           Arrs/Tup must have the same number of entries
//     FALSE: "subset check" for Tab/OTab to see if all *keys* of prototype
//              are in instance  
//           Arrs/Tup must have at least number of entries
//  type_match specifies how to match the types of *values* (see enum above)
//     Tabs/OTab how to type match the *values* 
//     Arrs/Tup specify how to match the values therein
//  throw_exception_with_message: where there's an error ...
//     FALSE: either return false (very cheap)
//     TRUE:  throw an exception with an informative error message (expensive)
//            (usually only set this to TRUE during debugging)
inline bool Conforms (const Val& instance, const Val& prototype,
		      bool exact_structure, 
		      Conform_e type_match,
		      bool throw_exception_with_informative_message)
{
  if (prototype.tag=='Z') return true; // No need for match
  if (instance.tag=='a' && type_match==LOOSE_STRING_MATCH) {
    return true;
  }
  else if (instance.tag=='o' || instance.tag=='t') {
    return TableMatch_(instance, prototype, exact_structure, type_match,
		       throw_exception_with_informative_message);
  } else if (instance.tag=='u' || instance.tag=='n') {
    return ArrayMatch_(instance, prototype, exact_structure, type_match,
		       throw_exception_with_informative_message);
  } else {
    return PrimitiveMatch_(instance, prototype, exact_structure, type_match,
			   throw_exception_with_informative_message);
  }
}


OC_END_NAMESPACE

#define OCCONFORMS_H_
#endif // OCCONFORMS_H_
